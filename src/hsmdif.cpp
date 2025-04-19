#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>
#include <grpcpp/grpcpp.h>
#include <boost/program_options.hpp>
#include <thread>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <shared_mutex>

// Include generated gRPC headers
#include "inference_service.grpc.pb.h"

// Include llama.cpp headers
#include "llama.cpp/llama.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::InferenceService;
using inference::InferenceRequest;
using inference::InferenceResponse;

namespace po = boost::program_options;
using json = nlohmann::json;

// Cache for tokenized prompts
std::shared_mutex token_cache_mutex;
std::unordered_map<std::string, std::vector<llama_token>> token_cache;

// Tokenizer class using llama.cpp
class Tokenizer {
public:
    Tokenizer(llama_model* model) : model_(model) {}

    std::vector<llama_token> tokenize(const std::string& text, int max_tokens) {
        // Check cache first
        {
            std::shared_lock lock(token_cache_mutex);
            auto it = token_cache.find(text);
            if (it != token_cache.end()) {
                auto tokens = it->second;
                if (tokens.size() > static_cast<size_t>(max_tokens)) {
                    tokens.resize(max_tokens);
                }
                return tokens;
            }
        }

        // Tokenize and cache
        std::vector<llama_token> tokens(max_tokens);
        int n_tokens = llama_tokenize(model_, text.c_str(), tokens.data(), max_tokens, true, nullptr);
        if (n_tokens < 0) {
            throw std::runtime_error("Tokenization failed");
        }
        tokens.resize(n_tokens);

        {
            std::unique_lock lock(token_cache_mutex);
            token_cache[text] = tokens;
        }

        return tokens;
    }

    std::string detokenize(const std::vector<llama_token>& tokens, llama_context* ctx) {
        std::string output;
        char buf[128];
        for (const auto& token : tokens) {
            int len = llama_token_to_piece(ctx, token, buf, sizeof(buf), false);
            if (len <= 0) continue;
            output += std::string(buf, len);
        }
        return output;
    }

private:
    llama_model* model_;
};

// Device structure to store memory, speed, priority, and performance metrics
struct Device {
    std::string address;
    int port;
    size_t total_memory; // in MB
    size_t memfree; // in MB
    double speed; // tokens per second
    int priority; // Priority rank (lower is higher priority)
    double load; // Current CPU load (0.0 to 1.0)
    double avg_latency; // Average network latency (ms)
    int token_share; // Number of tokens to generate
    bool is_root; // Indicates if this device is the root node
};

// gRPC Inference Service Implementation
class InferenceServiceImpl final : public InferenceService::Service {
public:
    InferenceServiceImpl(llama_model* model, int nthreads, int max_seq_len)
        : model_(model), nthreads_(nthreads), max_seq_len_(max_seq_len) {
        ctx_ = llama_new_context_with_model(model, nullptr);
        if (!ctx_) {
            throw std::runtime_error("Failed to create llama context on worker");
        }
        llama_context_params params = llama_context_default_params();
        params.n_ctx = max_seq_len;
        params.n_threads = nthreads;
        llama_set_n_threads(ctx_, nthreads);
    }

    ~InferenceServiceImpl() {
        if (ctx_) {
            llama_free(ctx_);
        }
    }

    Status RunInference(ServerContext* context, const InferenceRequest* request, InferenceResponse* response) override {
        auto start = std::chrono::high_resolution_clock::now();

        // Deserialize input tokens and context
        std::vector<llama_token> tokens;
        const std::string& input_data = request->input_tokens();
        tokens.resize(input_data.size() / sizeof(llama_token));
        std::memcpy(tokens.data(), input_data.data(), input_data.size());

        int n_past = request->n_past();
        int tokens_to_generate = request->tokens_to_generate();
        int batch_size = request->batch_size();

        // Evaluate the input tokens
        for (size_t i = 0; i < tokens.size(); i += batch_size) {
            int current_batch = std::min(batch_size, static_cast<int>(tokens.size() - i));
            llama_eval(ctx_, tokens.data() + i, current_batch, n_past, nullptr);
            n_past += current_batch;
        }

        // Generate new tokens in parallel batches
        std::vector<llama_token> output_tokens;
        for (int i = 0; i < tokens_to_generate; i += batch_size) {
            int current_gen_batch = std::min(batch_size, tokens_to_generate - i);
            std::vector<llama_token> batch_tokens(current_gen_batch);
            std::vector<std::thread> threads;

            for (int j = 0; j < current_gen_batch; ++j) {
                threads.emplace_back([this, &batch_tokens, j, n_past, current_gen_batch]() {
                    llama_token new_token = llama_sample_top_k_top_p(ctx_, nullptr, 0, 5, 0.8, 1.0, 1.0, nullptr);
                    batch_tokens[j] = new_token;
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }

            // Evaluate the batch of generated tokens
            llama_eval(ctx_, batch_tokens.data(), current_gen_batch, n_past, nullptr);
            n_past += current_gen_batch;
            output_tokens.insert(output_tokens.end(), batch_tokens.begin(), batch_tokens.end());
        }

        // Serialize output tokens
        std::string output_data(output_tokens.size() * sizeof(llama_token), 0);
        std::memcpy(&output_data[0], output_tokens.data(), output_data.size());
        response->set_output_tokens(output_data);
        response->set_n_past(n_past);

        // Calculate latency
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        response->set_latency_ms(latency_ms);

        return Status::OK;
    }

private:
    llama_model* model_;
    llama_context* ctx_;
    int nthreads_;
    int max_seq_len_;
};

// HSM-DIF class for managing distributed inference
class HSMDIF {
public:
    HSMDIF(const std::string& model_path, std::vector<Device>& devices, 
           bool prioritize_by_memory, std::string quantize, int max_seq_len, int nthreads,
           long long num_parameters)
        : model_path_(model_path), devices_(devices), quantize_(quantize), 
          max_seq_len_(max_seq_len), nthreads_(nthreads), num_parameters_(num_parameters) {
        if (quantize_ != "4bit" && quantize_ != "8bit") {
            throw std::runtime_error("Invalid quantization level. Use 4bit or 8bit.");
        }
        if (num_parameters_ <= 0) {
            throw std::runtime_error("Number of parameters must be positive");
        }
        if (prioritize_by_memory) {
            prioritizeByMemory();
        } else {
            prioritizeByRank();
        }
        setRootNode();
        loadModel();
        tokenizer_ = std::make_unique<Tokenizer>(model_);
    }

    ~HSMDIF() {
        if (model_) {
            llama_free(model_);
        }
        if (ctx_) {
            llama_free(ctx_);
        }
    }

    std::string inference(const std::string& input, int max_tokens) {
        // Tokenize the input
        std::vector<llama_token> tokens = tokenizer_->tokenize(input, max_seq_len_);
        if (tokens.size() > static_cast<size_t>(max_seq_len_)) {
            tokens.resize(max_seq_len_);
            std::cout << "Prompt truncated to " << max_seq_len_ << " tokens." << std::endl;
        }

        // Update device metrics and assign token shares
        updateDeviceMetrics();
        assignTokenShares(max_tokens);

        // Check if the root device can handle all tokens
        Device* root_device = nullptr;
        for (auto& device : devices_) {
            if (device.is_root) {
                root_device = &device;
                break;
            }
        }
        if (!root_device) {
            throw std::runtime_error("Root device not found");
        }

        // If only the root device is used, run standalone
        bool only_root = true;
        for (const auto& device : devices_) {
            if (device.token_share > 0 && !device.is_root) {
                only_root = false;
                break;
            }
        }

        if (only_root && root_device->token_share == max_tokens) {
            return runStandalone(tokens, max_tokens);
        } else {
            return runDistributed(tokens, max_tokens);
        }
    }

    void updateDevices(const std::string& config_path) {
        devices_ = loadDevicesFromConfig(config_path);
        prioritizeByMemory();
        setRootNode();
        loadModel(); // Reload model and clients with updated device list
    }

private:
    std::string model_path_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::vector<Device> devices_;
    std::string quantize_;
    size_t total_memory_needed_;
    int max_seq_len_;
    int nthreads_;
    long long num_parameters_;
    std::vector<std::shared_ptr<inference::InferenceService::Stub>> clients_;
    std::vector<double> recent_latencies_; // Track recent network latencies

    void prioritizeByMemory() {
        // Sort devices by total memory in descending order
        std::sort(devices_.begin(), devices_.end(), [](const Device& a, const Device& b) {
            return a.total_memory > b.total_memory;
        });
        for (size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].priority = i;
        }
    }

    void prioritizeByRank() {
        // Use predefined priority but fall back to memory if priorities are equal
        std::sort(devices_.begin(), devices_.end(), [](const Device& a, const Device& b) {
            if (a.priority == b.priority) {
                return a.total_memory > b.total_memory;
            }
            return a.priority < b.priority;
        });
        for (size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].priority = i;
        }
    }

    void setRootNode() {
        // Set the highest-priority device as the root node
        if (devices_.empty()) {
            throw std::runtime_error("No devices available to set as root node");
        }
        for (auto& device : devices_) {
            device.is_root = false;
        }
        auto highest_priority_device = std::min_element(devices_.begin(), devices_.end(),
            [](const Device& a, const Device& b) { return a.priority < b.priority; });
        highest_priority_device->is_root = true;
        std::cout << "Root node set to " << highest_priority_device->address << ":" << highest_priority_device->port << std::endl;
    }

    void updateDeviceMetrics() {
        // Metrics are updated by run_hsmdif.py via SSH
    }

    void loadModel() {
        // Free existing model and context if they exist
        if (model_) {
            llama_free(model_);
            model_ = nullptr;
        }
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        clients_.clear();

        // Load the model using llama.cpp with quantization
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0; // CPU only

        model_ = llama_load_model_from_file(model_path_.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("Failed to load model from " + model_path_);
        }

        // Calculate memory requirements based on number of parameters
        double bytes_per_param = (quantize_ == "4bit") ? 0.5 : 1.0; // 8bit
        size_t model_size_mb = static_cast<size_t>((num_parameters_ * bytes_per_param) / (1024 * 1024)); // Convert bytes to MB
        size_t context_size_mb = (max_seq_len_ * 4 * 100) / (1024 * 1024) + 2000; // Rough estimate: 4 bytes/token/layer, ~100 layers, +2GB buffer
        total_memory_needed_ = model_size_mb + context_size_mb;
        std::cout << "Model memory: " << model_size_mb << "MB, Context memory: " << context_size_mb << "MB, Total: " << total_memory_needed_ << "MB" << std::endl;

        // Initialize gRPC clients for each device (except the root) with compression
        for (const auto& device : devices_) {
            if (device.is_root) continue; // Skip the root node
            std::string target = device.address + ":" + std::to_string(device.port);
            grpc::ChannelArguments args;
            args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP); // Enable compression
            auto channel = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
            clients_.push_back(InferenceService::NewStub(channel));
            const_cast<Device&>(device).avg_latency = 0.0; // Initialize latency
        }
    }

    void assignTokenShares(int max_tokens) {
        // Reset token shares
        for (auto& device : devices_) {
            device.token_share = 0;
        }

        // Sort devices by memory capacity
        std::vector<Device*> active_devices;
        for (auto& device : devices_) {
            active_devices.push_back(&device);
        }
        std::sort(active_devices.begin(), active_devices.end(), [](const Device* a, const Device* b) {
            return a->total_memory > b->total_memory;
        });

        // Incrementally add devices until memory requirement is met
        size_t total_available_memory = 0;
        std::vector<Device*> selected_devices;
        for (auto& device : active_devices) {
            total_available_memory += device->memfree;
            selected_devices.push_back(device);
            if (total_available_memory >= total_memory_needed_) {
                break; // Enough memory allocated
            }
        }

        if (total_available_memory < total_memory_needed_) {
            throw std::runtime_error("Not enough combined memory to load the model (" + 
                                     std::to_string(total_available_memory) + "MB available, " + 
                                     std::to_string(total_memory_needed_) + "MB needed)");
        }

        // Calculate total effective capacity (memory * speed) of selected devices
        double total_capacity = 0.0;
        for (const auto& device : selected_devices) {
            total_capacity += static_cast<double>(device->memfree) * device->speed;
        }

        // Distribute tokens proportionally to effective capacity (memory * speed)
        int remaining_tokens = max_tokens;
        for (auto& device : selected_devices) {
            double capacity_ratio = (static_cast<double>(device->memfree) * device->speed) / total_capacity;
            int tokens_for_device = static_cast<int>(max_tokens * capacity_ratio);
            if (tokens_for_device == 0 && remaining_tokens > 0) {
                tokens_for_device = 1;
            }
            device->token_share = tokens_for_device;
            remaining_tokens -= tokens_for_device;
        }

        // Distribute any remaining tokens to the fastest device
        if (remaining_tokens > 0) {
            auto fastest_device = std::max_element(selected_devices.begin(), selected_devices.end(),
                [](const Device* a, const Device* b) { return a->speed < b->speed; });
            (*fastest_device)->token_share += remaining_tokens;
        }

        // Adjust token shares based on recent latency
        if (!recent_latencies_.empty()) {
            double avg_latency = std::accumulate(recent_latencies_.begin(), recent_latencies_.end(), 0.0) / recent_latencies_.size();
            for (auto& device : selected_devices) {
                if (device.avg_latency > avg_latency * 1.5 && device->token_share > 1) {
                    int reduced_tokens = device->token_share / 2;
                    device->token_share -= reduced_tokens;
                    // Redistribute to the fastest device
                    auto fastest_device = std::max_element(selected_devices.begin(), selected_devices.end(),
                        [](const Device* a, const Device* b) { return a->speed < b->speed; });
                    (*fastest_device)->token_share += reduced_tokens;
                }
            }
        }

        // Log the allocation
        for (const auto& device : selected_devices) {
            std::cout << "Device " << device->address << ":" << device->port 
                      << " allocated " << device->token_share << " tokens (Memory: " 
                      << device->memfree << "MB, Speed: " << device->speed << " tokens/sec)" << std::endl;
        }
    }

    std::string runStandalone(const std::vector<llama_token>& tokens, int max_tokens) {
        Device* root_device = nullptr;
        for (auto& device : devices_) {
            if (device.is_root) {
                root_device = &device;
                break;
            }
        }
        if (!root_device) {
            throw std::runtime_error("Root device not found");
        }

        std::cout << "Running standalone on " << root_device->address << ":" << root_device->port << std::endl;

        ctx_ = llama_new_context_with_model(model_, nullptr);
        if (!ctx_) {
            throw std::runtime_error("Failed to create llama context");
        }

        llama_context_params params = llama_context_default_params();
        params.n_ctx = max_seq_len_;
        params.n_threads = nthreads_;
        llama_set_n_threads(ctx_, nthreads_);

        int optimal_batch_size = std::max(1, nthreads_ / 2); // Optimize batch size
        int n_past = 0;
        for (size_t i = 0; i < tokens.size(); i += optimal_batch_size) {
            int batch_size = std::min(optimal_batch_size, static_cast<int>(tokens.size() - i));
            llama_eval(ctx_, const_cast<llama_token*>(&tokens[i]), batch_size, n_past, nullptr);
            n_past += batch_size;
        }

        std::vector<llama_token> output_tokens;
        for (int i = 0; i < max_tokens; i += optimal_batch_size) {
            int current_gen_batch = std::min(optimal_batch_size, max_tokens - i);
            std::vector<llama_token> batch_tokens(current_gen_batch);
            std::vector<std::thread> threads;

            for (int j = 0; j < current_gen_batch; ++j) {
                threads.emplace_back([this, &batch_tokens, j, n_past, current_gen_batch]() {
                    llama_token new_token = llama_sample_top_k_top_p(ctx_, nullptr, 0, 5, 0.8, 1.0, 1.0, nullptr);
                    batch_tokens[j] = new_token;
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }

            llama_eval(ctx_, batch_tokens.data(), current_gen_batch, n_past, nullptr);
            n_past += current_gen_batch;
            output_tokens.insert(output_tokens.end(), batch_tokens.begin(), batch_tokens.end());
        }

        std::string output = tokenizer_->detokenize(output_tokens, ctx_);
        llama_free(ctx_);
        ctx_ = nullptr;
        return output;
    }

    std::string runDistributed(const std::vector<llama_token>& tokens, int max_tokens) {
        std::cout << "Running distributed inference across devices" << std::endl;

        // Serialize initial tokens
        std::string input_data(tokens.size() * sizeof(llama_token), 0);
        std::memcpy(&input_data[0], tokens.data(), input_data.size());

        std::vector<llama_token> current_tokens = tokens;
        std::vector<llama_token> final_output_tokens;
        int n_past = 0;
        size_t client_index = 0;

        int optimal_batch_size = std::max(1, nthreads_ / 2); // Optimize batch size

        // Process tokens in a pipeline across devices
        for (size_t i = 0; i < devices_.size(); ++i) {
            auto& device = devices_[i];
            if (device.token_share == 0) continue;

            std::cout << "Device " << device.address << ":" << device->port << " generating " << device->token_share << " tokens" << std::endl;

            InferenceRequest request;
            request.set_input_tokens(input_data);
            request.set_n_past(n_past);
            request.set_tokens_to_generate(device->token_share);
            request.set_batch_size(optimal_batch_size);

            InferenceResponse response;
            if (device.is_root) {
                // Root node handles its share locally
                ctx_ = llama_new_context_with_model(model_, nullptr);
                if (!ctx_) {
                    throw std::runtime_error("Failed to create llama context on root node");
                }
                llama_context_params params = llama_context_default_params();
                params.n_ctx = max_seq_len_;
                params.n_threads = nthreads_;
                llama_set_n_threads(ctx_, nthreads_);

                for (size_t j = 0; j < current_tokens.size(); j += optimal_batch_size) {
                    int batch_size = std::min(optimal_batch_size, static_cast<int>(current_tokens.size() - j));
                    llama_eval(ctx_, current_tokens.data() + j, batch_size, n_past, nullptr);
                    n_past += batch_size;
                }

                std::vector<llama_token> output_tokens;
                for (int j = 0; j < device.token_share; j += optimal_batch_size) {
                    int current_gen_batch = std::min(optimal_batch_size, device.token_share - j);
                    std::vector<llama_token> batch_tokens(current_gen_batch);
                    std::vector<std::thread> threads;

                    for (int k = 0; k < current_gen_batch; ++k) {
                        threads.emplace_back([this, &batch_tokens, k, n_past, current_gen_batch]() {
                            llama_token new_token = llama_sample_top_k_top_p(ctx_, nullptr, 0, 5, 0.8, 1.0, 1.0, nullptr);
                            batch_tokens[k] = new_token;
                        });
                    }

                    for (auto& thread : threads) {
                        thread.join();
                    }

                    llama_eval(ctx_, batch_tokens.data(), current_gen_batch, n_past, nullptr);
                    n_past += current_gen_batch;
                    output_tokens.insert(output_tokens.end(), batch_tokens.begin(), batch_tokens.end());
                }

                current_tokens = output_tokens;
                final_output_tokens.insert(final_output_tokens.end(), current_tokens.begin(), current_tokens.end());

                llama_free(ctx_);
                ctx_ = nullptr;

                // Update input_data for the next device
                input_data.resize(current_tokens.size() * sizeof(llama_token));
                std::memcpy(&input_data[0], current_tokens.data(), input_data.size());
            } else {
                // Worker node handles its share via gRPC
                if (client_index >= clients_.size()) {
                    throw std::runtime_error("Client index out of range for device " + device.address);
                }
                auto& client = clients_[client_index++];
                grpc::ClientContext context;
                auto start = std::chrono::high_resolution_clock::now();
                Status status = client->RunInference(&context, request, &response);
                auto end = std::chrono::high_resolution_clock::now();
                double latency = std::chrono::duration<double, std::milli>(end - start).count();

                if (!status.ok()) {
                    throw std::runtime_error("gRPC inference failed on " + device.address + ": " + status.error_message());
                }

                // Update latency
                device.avg_latency = (device.avg_latency * 0.9) + (latency * 0.1);
                recent_latencies_.push_back(latency);
                if (recent_latencies_.size() > 10) recent_latencies_.erase(recent_latencies_.begin());

                // Deserialize output tokens
                const std::string& output_data = response.output_tokens();
                current_tokens.resize(output_data.size() / sizeof(llama_token));
                std::memcpy(current_tokens.data(), output_data.data(), output_data.size());
                n_past = response.n_past();

                final_output_tokens.insert(final_output_tokens.end(), current_tokens.begin(), current_tokens.end());
                input_data = output_data;
            }
        }

        return tokenizer_->detokenize(final_output_tokens, nullptr);
    }
};

// Load devices from a JSON configuration file
std::vector<Device> loadDevicesFromConfig(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }

    json config;
    file >> config;
    std::vector<Device> devices;

    for (const auto& device_json : config["devices"]) {
        Device device;
        device.address = device_json["address"];
        device.port = device_json["port"];
        device.total_memory = device_json["total_memory"];
        device.memfree = device_json["memfree"];
        device.speed = device_json["speed"];
        device.priority = 0;
        device.load = device_json.contains("load") ? device_json["load"] : 0.0;
        device.avg_latency = 0.0;
        device.token_share = 0;
        device.is_root = false;
        devices.push_back(device);
    }

    return devices;
}

// Check if a cached device profile exists
bool hasCachedProfile(const std::string& workers) {
    std::string cache_file = "device_cache_" + std::to_string(std::hash<std::string>{}(workers)) + ".json";
    return std::filesystem::exists(cache_file);
}

// Cache the device profile
void cacheProfile(const std::string& workers) {
    std::string cache_file = "device_cache_" + std::to_string(std::hash<std::string>{}(workers)) + ".json";
    std::ifstream src("temp_devices.json");
    std::ofstream dst(cache_file);
    dst << src.rdbuf();
}

// Run the gRPC server on a worker node
void RunServer(const std::string& model_path, int port, int nthreads, int max_seq_len) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only
    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("Failed to load model on worker");
    }

    InferenceServiceImpl service(model, nthreads, max_seq_len);
    ServerBuilder builder;
    builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_GZIP); // Enable compression
    std::string server_address = "0.0.0.0:" + std::to_string(port);
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Worker server listening on " << server_address << std::endl;
    server->Wait();
    llama_free(model);
}

int main(int argc, char* argv[]) {
    try {
        std::string model_path;
        std::string tokenizer_path;
        std::string workers;
        std::string priority;
        std::string config_path;
        std::string prompt = "Hello, how are you?";
        bool prioritize_by_memory = false;
        bool worker_mode = false;
        int port = 9999;
        std::string quantize = "8bit"; // Default to 8-bit, can be overridden
        int max_seq_len = 4096;
        int max_tokens = 128;
        int nthreads = 4;
        long long num_parameters = 3000000000; // Default to 3B parameters

        po::options_description desc("HSM-DIF Options");
        desc.add_options()
            ("help", "Produce help message")
            ("worker", po::bool_switch(&worker_mode), "Run in worker mode")
            ("port", po::value<int>(&port)->default_value(9999), "Port for worker server")
            ("model", po::value<std::string>(&model_path)->required(), "Path to model file")
            ("tokenizer", po::value<std::string>(&tokenizer_path), "Path to tokenizer file (required for main node)")
            ("workers", po::value<std::string>(&workers), "Comma-separated list of workers (address:port)")
            ("priority", po::value<std::string>(&priority), "Comma-separated list of device priorities")
            ("config", po::value<std::string>(&config_path), "Path to JSON config file for devices")
            ("prompt", po::value<std::string>(&prompt), "Prompt for inference")
            ("max-seq-len", po::value<int>(&max_seq_len)->default_value(4096), "Maximum sequence length (in tokens)")
            ("max-tokens", po::value<int>(&max_tokens)->default_value(128), "Maximum tokens to generate")
            ("nthreads", po::value<int>(&nthreads)->default_value(4), "Number of threads")
            ("prioritize-by-memory", po::bool_switch(&prioritize_by_memory), "Prioritize devices by memory capacity")
            ("quantize", po::value<std::string>(&quantize)->default_value("8bit"), "Quantization level (4bit, 8bit)")
            ("num-parameters", po::value<long long>(&num_parameters)->default_value(3000000000), "Number of model parameters");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);

        if (worker_mode) {
            // Run as a worker node
            RunServer(model_path, port, nthreads, max_seq_len);
            return 0;
        }

        // Run as main node
        if (!vm.count("tokenizer")) {
            throw std::runtime_error("Tokenizer path is required for main node");
        }
        if (!vm.count("workers") && !vm.count("config")) {
            throw std::runtime_error("Must specify either --workers or --config");
        }

        // Load devices
        std::vector<Device> devices;
        if (vm.count("config")) {
            devices = loadDevicesFromConfig(config_path);
        } else if (vm.count("workers")) {
            if (!hasCachedProfile(workers)) {
                std::string profile_cmd = "python scripts/profile_devices.py --devices \"" + workers + "\" > temp_devices.json";
                int ret = std::system(profile_cmd.c_str());
                if (ret != 0) {
                    throw std::runtime_error("Failed to profile devices");
                }
                cacheProfile(workers);
            } else {
                std::string cache_file = "device_cache_" + std::to_string(std::hash<std::string>{}(workers)) + ".json";
                std::ifstream src(cache_file);
                std::ofstream dst("temp_devices.json");
                dst << src.rdbuf();
            }
            devices = loadDevicesFromConfig("temp_devices.json");
        }

        if (!priority.empty()) {
            std::vector<std::string> priority_list = split(priority, ',');
            for (size_t i = 0; i < devices.size(); ++i) {
                for (size_t j = 0; j < priority_list.size(); ++j) {
                    if (devices[i].address == priority_list[j]) {
                        devices[i].priority = j;
                        break;
                    }
                }
            }
        }

        HSMDIF hsmdif(model_path, devices, prioritize_by_memory, quantize, max_seq_len, nthreads, num_parameters);
        auto start = std::chrono::high_resolution_clock::now();
        std::string output = hsmdif.inference(prompt, max_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        double tokens_per_second = max_tokens / duration;
        std::cout << "Output: " << output << std::endl;
        std::cout << "Inference speed: " << tokens_per_second << " tokens/second" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// Utility function to split strings
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}