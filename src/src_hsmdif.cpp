#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <zlib.h>
#include "model.h"
#include "network.h"
#include "utils.h"

namespace po = boost::program_options;
using boost::asio::ip::tcp;

// Device structure to store memory, speed, and priority
struct Device {
    std::string address;
    int port;
    size_t total_memory; // in MB
    size_t memfree; // in MB
    double speed; // tokens per second
    int priority; // Priority rank (lower is higher priority)
};

// Model slice structure
struct ModelSlice {
    int start_layer;
    int end_layer;
    Device* device;
};

// HSM-DIF class for managing distributed inference
class HSMDIF {
public:
    HSMDIF(const std::string& model_path, const std::vector<Device>& devices, bool prioritize_by_memory, bool quantize_8bit)
        : model_(model_path), devices_(devices), quantize_8bit_(quantize_8bit) {
        if (prioritize_by_memory) {
            prioritizeByMemory();
        } else {
            prioritizeByRank();
        }
        loadModel();
    }

    void inference(const std::string& input) {
        // Check if model fits on first-priority device
        if (canRunOnSingleDevice()) {
            runStandalone(input);
        } else {
            runDistributed(input);
        }
    }

private:
    Model model_;
    std::vector<Device> devices_;
    std::vector<ModelSlice> slices_;
    bool quantize_8bit_;
    size_t total_memory_needed_;

    void prioritizeByMemory() {
        // Sort devices by memory capacity, prioritizing root PC (devices[0]) if tied
        std::sort(devices_.begin(), devices_.end(), [&](const Device& a, const Device& b) {
            if (a.total_memory == b.total_memory) {
                return &a == &devices_[0]; // Prioritize root PC
            }
            return a.total_memory > b.total_memory;
        });
        for (size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].priority = i;
        }
    }

    void prioritizeByRank() {
        // Already sorted by priority in input
        for (size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].priority = i;
        }
    }

    void loadModel() {
        // Estimate model memory requirements (simplified)
        size_t model_size_mb = model_.num_parameters() * (quantize_8bit_ ? 1 : 2); // FP16 or 8-bit
        total_memory_needed_ = model_size_mb;

        // Assign layers to devices based on prioritization
        size_t total_layers = model_.num_layers();
        size_t assigned_layers = 0;
        for (auto& device : devices_) {
            if (assigned_layers == total_layers) break;

            size_t device_capacity_mb = device.memfree - 500; // Reserve 500 MB for safety
            size_t layers_for_device = std::min(
                total_layers - assigned_layers,
                static_cast<size_t>(device_capacity_mb * total_layers / total_memory_needed_)
            );

            if (layers_for_device > 0) {
                ModelSlice slice;
                slice.start_layer = assigned_layers;
                slice.end_layer = assigned_layers + layers_for_device - 1;
                slice.device = &device;
                slices_.push_back(slice);
                assigned_layers += layers_for_device;
            }
        }

        if (assigned_layers < total_layers) {
            throw std::runtime_error("Not enough memory across devices to load model");
        }

        // Load model weights on each device
        for (const auto& slice : slices_) {
            sendWeightsToDevice(slice.device->address, slice.device->port, slice.start_layer, slice.end_layer);
        }
    }

    bool canRunOnSingleDevice() {
        // Check if the first-priority device can handle the entire model
        Device& first_device = devices_[0];
        return total_memory_needed_ <= first_device.memfree - 500;
    }

    void runStandalone(const std::string& input) {
        Device& device = devices_[0];
        std::cout << "Running standalone on " << device.address << ":" << device.port << std::endl;

        tcp::socket socket(io_context_);
        tcp::resolver resolver(io_context_);
        boost::asio::connect(socket, resolver.resolve(device.address, std::to_string(device.port)));

        // Send input and receive output
        send(socket, input);
        std::string output = receive(socket);
        std::cout << "Output: " << output << std::endl;
    }

    void runDistributed(const std::string& input) {
        std::cout << "Running distributed inference across " << slices_.size() << " devices" << std::endl;

        std::vector<std::unique_ptr<tcp::socket>> sockets;
        for (const auto& slice : slices_) {
            auto socket = std::make_unique<tcp::socket>(io_context_);
            tcp::resolver resolver(io_context_);
            boost::asio::connect(*socket, resolver.resolve(slice.device->address, std::to_string(slice.device->port)));
            sockets.push_back(std::move(socket));
        }

        // Asynchronous inference with layer-wise scheduling
        std::string current_input = input;
        for (size_t i = 0; i < slices_.size(); ++i) {
            auto& socket = *sockets[i];
            sendAsync(socket, current_input, [this, &current_input, &socket](const std::string& output) {
                current_input = decompress(output); // Decompress output
            });
        }

        io_context_.run();
        std::cout << "Final Output: " << current_input << std::endl;
    }

    void sendWeightsToDevice(const std::string& address, int port, int start_layer, int end_layer) {
        tcp::socket socket(io_context_);
        tcp::resolver resolver(io_context_);
        boost::asio::connect(socket, resolver.resolve(address, std::to_string(port)));

        // Serialize and send weights for the assigned layers
        std::string weights = model_.serialize_weights(start_layer, end_layer);
        std::string compressed_weights = compress(weights);
        send(socket, compressed_weights);
    }

    std::string compress(const std::string& data) {
        z_stream zs;
        zs.zalloc = Z_NULL;
        zs.zfree = Z_NULL;
        zs.opaque = Z_NULL;
        deflateInit(&zs, Z_DEFAULT_COMPRESSION);

        zs.avail_in = data.size();
        zs.next_in = (Bytef*)data.data();
        std::string compressed;
        char buffer[8192];
        do {
            zs.avail_out = sizeof(buffer);
            zs.next_out = (Bytef*)buffer;
            deflate(&zs, Z_FINISH);
            compressed.append(buffer, sizeof(buffer) - zs.avail_out);
        } while (zs.avail_out == 0);

        deflateEnd(&zs);
        return compressed;
    }

    std::string decompress(const std::string& data) {
        z_stream zs;
        zs.zalloc = Z_NULL;
        zs.zfree = Z_NULL;
        zs.opaque = Z_NULL;
        inflateInit(&zs);

        zs.avail_in = data.size();
        zs.next_in = (Bytef*)data.data();
        std::string decompressed;
        char buffer[8192];
        do {
            zs.avail_out = sizeof(buffer);
            zs.next_out = (Bytef*)buffer;
            inflate(&zs, Z_NO_FLUSH);
            decompressed.append(buffer, sizeof(buffer) - zs.avail_out);
        } while (zs.avail_out == 0);

        inflateEnd(&zs);
        return decompressed;
    }

    void send(tcp::socket& socket, const std::string& data) {
        boost::asio::write(socket, boost::asio::buffer(data));
    }

    void sendAsync(tcp::socket& socket, const std::string& data, std::function<void(std::string)> callback) {
        std::string compressed_data = compress(data);
        boost::asio::async_write(socket, boost::asio::buffer(compressed_data),
            [this, &socket, callback](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    receiveAsync(socket, callback);
                }
            });
    }

    std::string receive(tcp::socket& socket) {
        boost::asio::streambuf buffer;
        boost::asio::read_until(socket, buffer, "\n");
        std::istream is(&buffer);
        std::string data;
        std::getline(is, data);
        return decompress(data);
    }

    void receiveAsync(tcp::socket& socket, std::function<void(std::string)> callback) {
        auto buffer = std::make_shared<boost::asio::streambuf>();
        boost::asio::async_read_until(socket, *buffer, "\n",
            [this, buffer, callback](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    std::istream is(buffer.get());
                    std::string data;
                    std::getline(is, data);
                    callback(decompress(data));
                }
            });
    }

    boost::asio::io_context io_context_;
};

int main(int argc, char* argv[]) {
    try {
        std::string model_path;
        std::string workers;
        std::string priority;
        bool prioritize_by_memory = false;
        bool quantize_8bit = false;

        po::options_description desc("HSM-DIF Options");
        desc.add_options()
            ("help", "Produce help message")
            ("model", po::value<std::string>(&model_path)->required(), "Path to model file")
            ("workers", po::value<std::string>(&workers)->required(), "Comma-separated list of workers (address:port)")
            ("priority", po::value<std::string>(&priority), "Comma-separated list of device priorities")
            ("prioritize-by-memory", po::bool_switch(&prioritize_by_memory), "Prioritize devices by memory capacity")
            ("quantize", po::value<std::string>()->default_value("none"), "Quantization level (8bit, none)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);

        if (vm["quantize"].as<std::string>() == "8bit") {
            quantize_8bit = true;
        }

        // Parse devices from workers string
        std::vector<Device> devices;
        std::vector<std::string> worker_list = split(workers, ',');
        for (const auto& worker : worker_list) {
            auto parts = split(worker, ':');
            Device device;
            device.address = parts[0];
            device.port = std::stoi(parts[1]);
            // Dummy values; will be updated by profiling script
            device.total_memory = 16000; // MB
            device.memfree = 8000; // MB
            device.speed = 10.0; // tokens/second
            devices.push_back(device);
        }

        // Update priorities based on input or memory
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

        HSMDIF hsmdif(model_path, devices, prioritize_by_memory, quantize_8bit);
        hsmdif.inference("Hello, how are you?");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}