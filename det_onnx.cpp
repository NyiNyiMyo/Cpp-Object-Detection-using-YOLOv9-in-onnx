#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <numeric>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

// Simple detection struct
struct Det {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

// IoU
float IoU(const Det& a, const Det& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6f);
}

// Class-aware NMS -> returns indices to keep
std::vector<int> NMS_class_aware(const std::vector<Det>& dets, float iou_thresh) {
    std::vector<int> idxs(dets.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    // sort descending by score
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });

    std::vector<int> keep;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t _i = 0; _i < idxs.size(); ++_i) {
        int i = idxs[_i];
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < idxs.size(); ++_j) {
            int j = idxs[_j];
            if (suppressed[j]) continue;
            if (dets[i].cls != dets[j].cls) continue; // keep across classes
            if (IoU(dets[i], dets[j]) > iou_thresh) suppressed[j] = true;
        }
    }
    return keep;
}

int main() {
    try {
        // -------------------
        // Settings (change as needed)
        // -------------------
        const std::wstring model_path = L"v9-s_mit.onnx"; // <- set your model path
        const std::string images_dir = "animals-tests";    // <- set your images folder
        const int input_w = 640;    
        const int input_h = 480;    
        const float conf_thresh = 0.25f;
        const float iou_thresh = 0.4f;
        const int max_images = 8;

        // COCO classes (same as you had)
        std::vector<std::string> classes = {
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
            "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
            "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
            "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
            "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
            "clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };

        // -------------------
        // Initialize ONNX Runtime
        // -------------------
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv9");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        // you can append CUDA EP here if desired similar to your C# (requires linking/compilation flags)
        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // -------------------
        // Load image paths
        // -------------------
        std::vector<std::string> all_images;
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            if (entry.is_regular_file()) all_images.push_back(entry.path().string());
        }
        if (all_images.empty()) { std::cerr << "No images found in " << images_dir << "\n"; return -1; }

        std::shuffle(all_images.begin(), all_images.end(), std::mt19937{ std::random_device{}() });
        if ((int)all_images.size() > max_images) all_images.resize(max_images);

        // Input / output names
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        const char* output_name = output_name_ptr.get();

        // -------------------
        // Inference loop
        // -------------------
        std::vector<cv::Mat> vis_images;
        for (const auto& image_path : all_images) {
            cv::Mat img_bgr = cv::imread(image_path);
            if (img_bgr.empty()) { std::cerr << "Failed to read: " << image_path << "\n"; continue; }

            int orig_w = img_bgr.cols;
            int orig_h = img_bgr.rows;

            // convert BGR -> RGB (as in your C#)
            cv::Mat img;
            cv::cvtColor(img_bgr, img, cv::COLOR_BGR2RGB);

            // Resize to input size (note: this stretches; same as your C# Resize)
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(input_w, input_h));
            resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

            // HWC -> CHW float vector
            std::vector<float> input_tensor_values(1 * 3 * input_h * input_w);
            size_t idx = 0;
            // order: channel major (c, y, x) matching your C# loop
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < input_h; ++y) {
                    for (int x = 0; x < input_w; ++x) {
                        cv::Vec3f px = resized.at<cv::Vec3f>(y, x);
                        input_tensor_values[idx++] = px[c];
                    }
                }
            }

            std::array<int64_t, 4> input_dims = { 1, 3, input_h, input_w };
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size()
            );

            // Run
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, &output_name, 1);
            Ort::Value& out_tensor = output_tensors.front();

            // get output buffer and length
            size_t out_elem_count = out_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
            float* out_data = out_tensor.GetTensorMutableData<float>();

            if (out_elem_count % 6 != 0) {
                std::cerr << "Unexpected output shape: elements not divisible by 6\n";
                continue;
            }
            size_t num_rows = out_elem_count / 6;

            // Collect detections (in model coordinate space)
            std::vector<Det> dets;
            dets.reserve(num_rows);
            for (size_t r = 0; r < num_rows; ++r) {
                float x1 = out_data[r * 6 + 0];
                float y1 = out_data[r * 6 + 1];
                float x2 = out_data[r * 6 + 2];
                float y2 = out_data[r * 6 + 3];
                float score = out_data[r * 6 + 4];
                float cls_f = out_data[r * 6 + 5];
                int cls = static_cast<int>(std::round(cls_f)); // ensure int

                if (score < conf_thresh) continue;

                // push after scaling later (we keep in model space for now)
                Det d; d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2; d.score = score; d.cls = cls;
                dets.push_back(d);
            }

            // If no detections continue
            if (dets.empty()) {
                vis_images.push_back(img_bgr); // original bgr for visualization
                continue;
            }

            // NMS (class-aware)
            std::vector<int> keep_idx = NMS_class_aware(dets, iou_thresh);

            // Draw boxes: must rescale from input size -> original image size
            float scale_x = static_cast<float>(orig_w) / input_w;
            float scale_y = static_cast<float>(orig_h) / input_h;

            cv::Mat vis = img_bgr.clone(); // draw on original BGR
            for (int k : keep_idx) {
                Det d = dets[k];
                int x1 = static_cast<int>(d.x1 * scale_x);
                int y1 = static_cast<int>(d.y1 * scale_y);
                int x2 = static_cast<int>(d.x2 * scale_x);
                int y2 = static_cast<int>(d.y2 * scale_y);
                int cls_id = d.cls;
                float conf = d.score;

                if (cls_id < 0 || cls_id >= (int)classes.size()) continue;
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 3);
                std::stringstream ss;
                ss << classes[cls_id] << " " << std::fixed << std::setprecision(2) << conf;
                cv::putText(vis, ss.str(), cv::Point(x1, std::max(0, y1 - 15)), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(0, 255, 0), 3);
            }

            vis_images.push_back(vis);
        } // for each image

        // -------------------
        // Display grid (2x4)
        // -------------------
        if (vis_images.empty()) {
            std::cerr << "No images to display!\n";
            return -1;
        }

        int rows = 2, cols = 4;
        int cell_w = 256, cell_h = 256;
        cv::Mat grid(rows * cell_h, cols * cell_w, vis_images[0].type(), cv::Scalar(0, 0, 0));

        for (size_t i = 0; i < vis_images.size(); ++i) {
            int r = i / cols;
            int c = i % cols;

            cv::Mat img = vis_images[i];
            cv::Mat resized;
            float scale = std::min((float)cell_w / img.cols, (float)cell_h / img.rows);
            int new_w = (int)(img.cols * scale);
            int new_h = (int)(img.rows * scale);

            cv::resize(img, resized, cv::Size(new_w, new_h));

            cv::Mat canvas(cell_h, cell_w, img.type(), cv::Scalar(0, 0, 0));
            int offset_x = (cell_w - new_w) / 2;
            int offset_y = (cell_h - new_h) / 2;
            resized.copyTo(canvas(cv::Rect(offset_x, offset_y, new_w, new_h)));

            cv::Rect roi(c * cell_w, r * cell_h, cell_w, cell_h);
            canvas.copyTo(grid(roi));
        }

        cv::namedWindow("YOLOv9 ONNX Grid", cv::WINDOW_AUTOSIZE);
        cv::imshow("YOLOv9 ONNX Grid", grid);
        cv::waitKey(0);
        cv::destroyAllWindows();

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
