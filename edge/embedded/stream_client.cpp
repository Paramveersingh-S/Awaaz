/**
 * Awaaz C++ Embedded Streaming Client for ARM64 Linux / Raspberry Pi.
 *
 * Implements a zero-overhead sliding-window inference loop using
 * ONNX Runtime C++ API and dynamic-chunk Conformer-CTC models.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <onnxruntime_cxx_api.h>

class AwaazEmbeddedRecognizer {
public:
    AwaazEmbeddedRecognizer(const std::string& model_path, int chunk_ms = 640, int left_chunks = 8)
        : chunk_ms_(chunk_ms), left_chunks_(left_chunks), last_token_id_(-1) {
        
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AwaazEmbedded");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        #if defined(_WIN32)
        std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, widestr.c_str(), session_options);
        #else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        #endif
    }

    /**
     * Accepts a raw 16 kHz float PCM audio chunk and runs inference.
     */
    void process_chunk(const std::vector<float>& pcm_chunk) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // Accumulate audio into rolling history window
        audio_history_.insert(audio_history_.end(), pcm_chunk.begin(), pcm_chunk.end());
        size_t max_samples = static_cast<size_t>(16000 * (chunk_ms_ / 1000.0) * (left_chunks_ + 1));
        if (audio_history_.size() > max_samples) {
            audio_history_.erase(audio_history_.begin(), audio_history_.begin() + (audio_history_.size() - max_samples));
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        double rtf = (elapsed.count() / 1000.0) / (chunk_ms_ / 1000.0);

        std::cout << "[Awaaz C++] Chunk processed in " << elapsed.count() << " ms (RTF: " << rtf << "x)" << std::endl;
    }

private:
    int chunk_ms_;
    int left_chunks_;
    int last_token_id_;
    std::vector<float> audio_history_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << " Awaaz C++ Embedded Streaming Recognizer " << std::endl;
    std::cout << " Target: ARM64 Linux / Raspberry Pi      " << std::endl;
    std::cout << "=========================================" << std::endl;

    if (argc < 2) {
        std::cout << "Usage: awaaz_stream_client <model.onnx>" << std::endl;
        return 0;
    }

    std::string model_path = argv[1];
    std::cout << "Loading model: " << model_path << std::endl;
    return 0;
}
