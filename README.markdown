# HSM-DIF LLaMA: Hybrid Speed-Memory-Aware Distributed Inference Framework

This repository implements the HSM-DIF framework for efficient distributed inference of large language models (LLMs) on heterogeneous clusters. It addresses bottlenecks in memory management, load balancing, communication latency, and device prioritization, while supporting flexible device configurations.

## Features

- Speed-memory-aware token-level pipeline parallelism across devices.
- Two prioritization options:
  - By memory capacity (`--prioritize-by-memory`).
  - By predefined device priority (`--priority`).
- Incremental device usage: combines devices’ memory until the model’s requirement is met, starting with the highest-priority device.
- Automatic root node selection: the highest-priority device (based on memory or custom priority) acts as the root node for coordination.
- Persistent model weights: weights remain loaded for multiple inferences in interactive mode.
- Dynamic device management: add, remove, or shuffle devices at runtime by updating the configuration.
- Support for arbitrary model sizes (e.g., 3B to 30B parameters) with memory estimation.
- Hybrid inference mode: standalone for small models, distributed for large models.
- Inactive memory management to reclaim unused memory safely.
- Adaptive load balancing based on device capabilities and current load.
- Optimized communication with gRPC for low-latency distributed inference.
- Support for 4-bit and 8-bit quantization.
- Flexible worker specification via command-line (IP:Port) or JSON config file.
- Single-command execution with an interactive prompt for user inputs.
- Support for any prompt length with configurable maximum sequence length (in tokens).
- Optimized for speed with multi-threading, tokenization caching, and device profile caching.
- Tokenizer support for proper prompt tokenization using `llama.cpp`.

## Setup

### Prerequisites

- **Operating System**: Linux/macOS (tested on Ubuntu 22.04 and macOS Ventura).
- **Dependencies**:
  - C++17 compiler (e.g., `g++` or `clang++`).
  - CMake (version 3.10 or higher).
  - Boost (version 1.65 or higher, for program options).
  - nlohmann_json (for JSON parsing).
  - gRPC and Protobuf (for distributed communication).
  - Python 3.8+ (for profiling, memory management, and wrapper script).
  - `llama.cpp` (for model inference and tokenization).

### Steps to Clone and Upload to GitHub

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/hsmdif-llama.git
   cd hsmdif-llama
   ```

   If you haven't created the repository yet, create it on GitHub and initialize locally:

   ```bash
   mkdir hsmdif-llama
   cd hsmdif-llama
   git init
   git remote add origin https://github.com/yourusername/hsmdif-llama.git
   ```

2. **Add All Files**:

   Copy the files provided (e.g., `src/hsmdif.cpp`, `run_hsmdif.py`, etc.) into the appropriate directories. The structure should look like:

   ```
   hsmdif-llama/
   ├── src/
   │   └── hsmdif.cpp
   ├── scripts/
   │   ├── inactive_memory_manager.py
   │   └── profile_devices.py
   ├── inference_service.proto
   ├── CMakeLists.txt
   ├── README.md
   ├── requirements.txt
   └── run_hsmdif.py
   ```

3. **Clone** `llama.cpp`:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

4. **Commit and Push to GitHub**:

   ```bash
   git add .
   git commit -m "Initial commit of HSM-DIF LLaMA project"
   git push origin main
   ```

### Installation on Devices

#### Device Information

- **Mac Mini (192.168.1.100)**: 16GB RAM, \~14GB free, macOS Ventura.
- **Raspberry Pi 1 (10.0.0.179)**: 16GB RAM, \~14GB free, Ubuntu 22.04.
- **Raspberry Pi 2 (10.0.0.124)**: 8GB RAM, \~7GB free, Ubuntu 22.04.
- **Raspberry Pi 3 (10.0.0.244)**: 4GB RAM, \~3.5GB free, Ubuntu 22.04.

#### Installation Steps

##### On Mac Mini (192.168.1.100)

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/hsmdif-llama.git
   cd hsmdif-llama
   ```

2. **Clone** `llama.cpp`:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

3. **Install Dependencies**:

   ```bash
   brew install cmake boost nlohmann-json grpc protobuf
   pip install -r requirements.txt
   ```

4. **Build the Project**:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   cd ..
   ```

5. **Open Port 9999**:

   ```bash
   sudo pfctl -ef <(echo "pass in on lo0 proto tcp from any to any port 9999")
   ```

6. **Setup SSH (for Passwordless Access)**:

   Generate an SSH key and copy it to all Raspberry Pis:

   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ssh-copy-id user@10.0.0.179
   ssh-copy-id user@10.0.0.124
   ssh-copy-id user@10.0.0.244
   ```

##### On Raspberry Pi 1 (10.0.0.179)

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/hsmdif-llama.git
   cd hsmdif-llama
   ```

2. **Clone** `llama.cpp`:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

3. **Install Dependencies**:

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake libboost-all-dev nlohmann-json3-dev libgrpc++-dev protobuf-compiler-grpc grpc-tools
   pip install -r requirements.txt
   ```

4. **Build the Project**:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   cd ..
   ```

5. **Open Ports 9999 and 22**:

   ```bash
   sudo ufw allow 9999/tcp
   sudo ufw allow 22/tcp
   sudo ufw enable
   ```

##### On Raspberry Pi 2 (10.0.0.124)

Repeat the same steps as for Raspberry Pi 1:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/hsmdif-llama.git
   cd hsmdif-llama
   ```

2. **Clone** `llama.cpp`:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

3. **Install Dependencies**:

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake libboost-all-dev nlohmann-json3-dev libgrpc++-dev protobuf-compiler-grpc grpc-tools
   pip install -r requirements.txt
   ```

4. **Build the Project**:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   cd ..
   ```

5. **Open Ports 9999 and 22**:

   ```bash
   sudo ufw allow 9999/tcp
   sudo ufw allow 22/tcp
   sudo ufw enable
   ```

##### On Raspberry Pi 3 (10.0.0.244)

Repeat the same steps as for Raspberry Pi 1:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/hsmdif-llama.git
   cd hsmdif-llama
   ```

2. **Clone** `llama.cpp`:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

3. **Install Dependencies**:

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake libboost-all-dev nlohmann-json3-dev libgrpc++-dev protobuf-compiler-grpc grpc-tools
   pip install -r requirements.txt
   ```

4. **Build the Project**:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   cd ..
   ```

5. **Open Ports 9999 and 22**:

   ```bash
   sudo ufw allow 9999/tcp
   sudo ufw allow 22/tcp
   sudo ufw enable
   ```

#### Model Preparation (on Mac Mini)

1. **Convert and Quantize the Model**:

   Assuming you have a 30B parameter model in Hugging Face format (e.g., at `/path/to/llama-30b`), convert it to GGUF and quantize it:

   ```bash
   cd hsmdif-llama/llama.cpp
   python convert_hf_to_gguf.py /path/to/llama-30b --outfile ../models/llama_30b.gguf
   ./llama-quantize ../models/llama_30b.gguf ../models/llama_30b_q4_0.gguf q4_0
   ./llama-quantize ../models/llama_30b.gguf ../models/llama_30b_q8_0.gguf q8_0
   ```

2. **Copy Models to All Devices**:

   Use `scp` to copy the GGUF files to each Raspberry Pi:

   ```bash
   scp models/llama_30b_q4_0.gguf user@10.0.0.179:~/hsmdif-llama/models/
   scp models/llama_30b_q8_0.gguf user@10.0.0.179:~/hsmdif-llama/models/
   scp models/llama_30b_q4_0.gguf user@10.0.0.124:~/hsmdif-llama/models/
   scp models/llama_30b_q8_0.gguf user@10.0.0.124:~/hsmdif-llama/models/
   scp models/llama_30b_q4_0.gguf user@10.0.0.244:~/hsmdif-llama/models/
   scp models/llama_30b_q8_0.gguf user@10.0.0.244:~/hsmdif-llama/models/
   ```

#### Create `devices.json` (on Mac Mini)

Create a `devices.json` file with your cluster configuration:

```json
{
  "devices": [
    {
      "address": "192.168.1.100",
      "port": 9999,
      "total_memory": 16000,
      "memfree": 14000,
      "speed": 10.0,
      "load": 0.1
    },
    {
      "address": "10.0.0.179",
      "port": 9999,
      "total_memory": 16000,
      "memfree": 14000,
      "speed": 5.0,
      "load": 0.2
    },
    {
      "address": "10.0.0.124",
      "port": 9999,
      "total_memory": 8000,
      "memfree": 7000,
      "speed": 4.0,
      "load": 0.3
    },
    {
      "address": "10.0.0.244",
      "port": 9999,
      "total_memory": 4000,
      "memfree": 3500,
      "speed": 3.0,
      "load": 0.4
    }
  ]
}
```

## Usage

### Start Worker Nodes

#### On Raspberry Pi 1 (10.0.0.179)

```bash
cd hsmdif-llama
./build/hsmdif --worker --model models/llama_30b_q4_0.gguf --port 9999 --nthreads 4
```

**Expected Output**:

```
Worker server listening on 0.0.0.0:9999
```

#### On Raspberry Pi 2 (10.0.0.124)

```bash
cd hsmdif-llama
./build/hsmdif --worker --model models/llama_30b_q4_0.gguf --port 9999 --nthreads 4
```

**Expected Output**:

```
Worker server listening on 0.0.0.0:9999
```

#### On Raspberry Pi 3 (10.0.0.244)

```bash
cd hsmdif-llama
./build/hsmdif --worker --model models/llama_30b_q4_0.gguf --port 9999 --nthreads 4
```

**Expected Output**:

```
Worker server listening on 0.0.0.0:9999
```

**Note**: The model weights are loaded on each worker and remain loaded as long as the gRPC server runs.

### Run Inference on Main Node (Mac Mini)

#### Non-Interactive Mode (Single Inference)

For a 30B model at 4-bit quantization, prioritizing by memory:

```bash
cd hsmdif-llama
python run_hsmdif.py --non-interactive \
  --model models/llama_30b_q4_0.gguf \
  --tokenizer models/llama_30b_q4_0.gguf \
  --config devices.json \
  --prioritize-by-memory \
  --prompt "Tell me about distributed inference." \
  --max-seq-len 4096 \
  --max-tokens 128 \
  --nthreads 4 \
  --quantize 4bit \
  --num-parameters 30000000000
```

**Expected Output**:

```
Profiling devices...
Updating device metrics...
Starting inactive memory manager on each device...
Started memory manager on 192.168.1.100
Started memory manager on 10.0.0.179
Started memory manager on 10.0.0.124
Started memory manager on 10.0.0.244
Running inference...
Root node set to 192.168.1.100:9999
Model memory: 15000MB, Context memory: 2000MB, Total: 17000MB
Device 192.168.1.100:9999 allocated 85 tokens (Memory: 14000MB, Speed: 10 tokens/sec
```