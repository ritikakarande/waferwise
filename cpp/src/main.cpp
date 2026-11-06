// waferwise_infer: classify wafer maps with OpenVINO and send uncertain ones to review.
//
// Reads PNG wafer maps (black = off wafer, grey = pass, white = fail), runs an exported model and
// applies the confidence gate calibrated in Python. Each wafer is either decided automatically or
// marked REVIEW for a person. Results go to a CSV file; a summary with timings goes to the console.
//
// Usage:
//   waferwise_infer --model wafer_cnn_int8.xml --gate gate.json --input <png or folder>
//                   [--output results.csv] [--batch 1] [--device CPU] [--threads N] [--timings timings.csv]
//                   [--seconds S]

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model, gate, input;
    std::string output = "results.csv";
    std::string device = "CPU";
    std::string timings;  // optional per-batch timing file, read by scripts/benchmark.py
    std::size_t batch = 1;
    int threads = 0;  // 0 keeps the OpenVINO default
    double seconds = 0.0;  // above 0: cycle through the wafers for this long, for benchmarks
};

struct BatchTiming {
    std::size_t wafers;
    double load_ms, infer_ms, total_ms;
};

struct Gate {
    float threshold = 1.0f;
    int image_size = 64;
    std::vector<std::string> classes;
};

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

Options parse_args(int argc, char** argv) {
    Options options;
    std::map<std::string, std::string*> text_flags = {
        {"--model", &options.model}, {"--gate", &options.gate},     {"--input", &options.input},
        {"--output", &options.output}, {"--device", &options.device}, {"--timings", &options.timings},
    };
    for (int i = 1; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
        if (flag == "--batch") {
            options.batch = std::stoul(argv[i + 1]);
        } else if (flag == "--threads") {
            options.threads = std::stoi(argv[i + 1]);
        } else if (flag == "--seconds") {
            options.seconds = std::stod(argv[i + 1]);
        } else if (text_flags.count(flag) != 0) {
            *text_flags[flag] = argv[i + 1];
        } else {
            throw std::runtime_error("Unknown option " + flag);
        }
    }
    if (options.model.empty() || options.gate.empty() || options.input.empty() || options.batch == 0) {
        throw std::runtime_error(
            "Usage: waferwise_infer --model <xml> --gate <json> --input <png or folder> "
            "[--output results.csv] [--batch 1] [--device CPU] [--threads N] [--timings timings.csv] [--seconds S]");
    }
    return options;
}

Gate load_gate(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened()) throw std::runtime_error("Cannot read gate file " + path);
    Gate gate;
    file["threshold"] >> gate.threshold;
    file["image_size"] >> gate.image_size;
    file["classes"] >> gate.classes;
    if (gate.classes.empty()) throw std::runtime_error("No classes listed in " + path);
    return gate;
}

std::vector<std::string> list_wafers(const std::string& input) {
    std::vector<std::string> paths;
    if (input.size() > 4 && input.compare(input.size() - 4, 4, ".png") == 0) {
        paths.push_back(input);
    } else {
        cv::glob(input + "/*.png", paths, false);
        std::sort(paths.begin(), paths.end());
    }
    if (paths.empty()) throw std::runtime_error("No PNG wafer maps found in " + input);
    return paths;
}

// Loads one wafer as a float image with the same values the model was trained on: 0, 0.5 and 1.
// INTER_NEAREST_EXACT matches the resize used in training; plain INTER_NEAREST does not.
cv::Mat load_wafer(const std::string& path, int size) {
    const cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Cannot read " + path);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(size, size), 0, 0, cv::INTER_NEAREST_EXACT);

    cv::Mat lookup(1, 256, CV_32F);
    for (int value = 0; value < 256; ++value) {
        lookup.at<float>(value) = value < 64 ? 0.0f : (value < 192 ? 0.5f : 1.0f);
    }
    cv::Mat wafer;
    cv::LUT(resized, lookup, wafer);
    return wafer;
}

void run(const Options& options) {
    const Gate gate = load_gate(options.gate);
    const std::vector<std::string> paths = list_wafers(options.input);
    const std::size_t size = static_cast<std::size_t>(gate.image_size);
    const std::size_t pixels = size * size;
    const std::size_t num_classes = gate.classes.size();

    ov::Core core;
    ov::AnyMap config;
    if (options.threads > 0) config.insert(ov::inference_num_threads(options.threads));
    ov::CompiledModel model = core.compile_model(options.model, options.device, config);
    ov::InferRequest request = model.create_infer_request();

    std::ofstream csv(options.output);
    if (!csv) throw std::runtime_error("Cannot write " + options.output);
    csv << "file,predicted,confidence,decision\n" << std::fixed << std::setprecision(6);

    std::vector<float> batch(options.batch * pixels);
    std::size_t processed = 0;
    std::size_t reviewed = 0;
    double load_seconds = 0.0;
    double infer_seconds = 0.0;
    std::vector<BatchTiming> timings;
    const Clock::time_point total_start = Clock::now();

    const bool timed_run = options.seconds > 0.0;
    for (std::size_t first = 0;; first += options.batch) {
        if (timed_run ? seconds_since(total_start) >= options.seconds : first >= paths.size()) break;
        const std::size_t count = timed_run ? options.batch : std::min(options.batch, paths.size() - first);

        const Clock::time_point load_start = Clock::now();
        for (std::size_t i = 0; i < count; ++i) {
            const cv::Mat wafer = load_wafer(paths[(first + i) % paths.size()], gate.image_size);
            std::copy_n(wafer.ptr<float>(), pixels, batch.begin() + static_cast<std::ptrdiff_t>(i * pixels));
        }
        const double load_time = seconds_since(load_start);
        load_seconds += load_time;

        const Clock::time_point infer_start = Clock::now();
        request.set_input_tensor(ov::Tensor(ov::element::f32, ov::Shape{count, 1, size, size}, batch.data()));
        request.infer();
        const double infer_time = seconds_since(infer_start);
        infer_seconds += infer_time;

        const float* probabilities = request.get_output_tensor().data<const float>();
        for (std::size_t i = 0; i < count; ++i) {
            const float* row = probabilities + i * num_classes;
            const std::size_t best = static_cast<std::size_t>(std::max_element(row, row + num_classes) - row);
            const bool automatic = row[best] >= gate.threshold;
            reviewed += automatic ? 0 : 1;
            csv << paths[(first + i) % paths.size()] << ',' << gate.classes[best] << ',' << row[best] << ','
                << (automatic ? "AUTO" : "REVIEW") << '\n';
        }
        processed += count;
        timings.push_back({count, 1000.0 * load_time, 1000.0 * infer_time, 1000.0 * seconds_since(load_start)});
    }

    if (!options.timings.empty()) {
        std::ofstream file(options.timings);
        if (!file) throw std::runtime_error("Cannot write " + options.timings);
        file << "wafers,load_ms,infer_ms,total_ms\n" << std::fixed << std::setprecision(6);
        for (const BatchTiming& t : timings) {
            file << t.wafers << ',' << t.load_ms << ',' << t.infer_ms << ',' << t.total_ms << '\n';
        }
    }

    const double total_seconds = seconds_since(total_start);
    const double wafers = static_cast<double>(processed);
    std::cout << "Wafers:      " << processed << " (batch size " << options.batch << ", " << options.device << ")\n"
              << "Automatic:   " << processed - reviewed << "\n"
              << "To review:   " << reviewed << " (" << std::fixed << std::setprecision(1)
              << 100.0 * static_cast<double>(reviewed) / wafers << "%, threshold " << std::setprecision(3)
              << gate.threshold << ")\n"
              << "Load:        " << std::setprecision(3) << 1000.0 * load_seconds / wafers << " ms per wafer\n"
              << "Inference:   " << 1000.0 * infer_seconds / wafers << " ms per wafer\n"
              << "Throughput:  " << std::setprecision(0) << wafers / total_seconds << " wafers/s\n"
              << "Results:     " << options.output << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
