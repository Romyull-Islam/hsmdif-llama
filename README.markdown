
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



### Installation on Devices

#### Device Information
- **Mac Mini (192.168.1.10)**: Main node, 16GB RAM, ~14GB free, macOS Ventura.
- **Raspberry Pi 1 (10.0.0.1)**: Worker node, 16GB RAM, ~14GB free, Ubuntu 22.04.
- **Raspberry Pi 2 (10.0.0.2)**: Worker node, 8GB RAM, ~7GB free, Ubuntu 22.04.
- **Raspberry Pi 3 (10.0.0.3)**: Worker node, 4GB RAM, ~3.5GB free, Ubuntu 22.04.

<<<<<<< HEAD
#### On Mac Mini (192.168.1.10)
=======
- **Mac Mini (192.168.1.1)**: 16GB RAM, \~14GB free, macOS Ventura.
- **Raspberry Pi 1 (10.0.0.1)**: 16GB RAM, \~14GB free, Ubuntu 22.04.
- **Raspberry Pi 2 (10.0.0.2)**: 8GB RAM, \~7GB free, Ubuntu 22.04.
- **Raspberry Pi 3 (10.0.0.3)**: 4GB RAM, \~3.5GB free, Ubuntu 22.04.

#### Installation Steps

##### On Mac Mini (192.168.1.100)

>>>>>>> ba728aaf3bf058ba4686e33f1e026180fd191b6a
1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Romyull-Islam/hsmdif-llama.git
   cd hsmdif-llama
   ```

2. **Clone `llama.cpp`**:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

3. **Install Dependencies**:

   ```bash
   brew install cmake boost nlohmann-json grpc protobuf
   pip install -r requirements.txt
   ```

   - **Note**: If Homebrew is not installed, install it first:
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
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

6. **Setup SSH for Passwordless Access**:

   Generate an SSH key and copy it to all Raspberry Pis:

   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ssh-copy-id user@10.0.0.1
   ssh-copy-id user@10.0.0.2
   ssh-copy-id user@10.0.0.3
   ```

<<<<<<< HEAD
   Replace `user` with the actual username on the Raspberry Pis. Alternatively, update `SSH_USERNAME` and `SSH_PASSWORD` in `run_hsmdif.py` and `scripts/profile_devices.py` with your credentials.
=======
##### On Raspberry Pi 1-3 (10.0.0.1)-10.0.0.3
>>>>>>> ba728aaf3bf058ba4686e33f1e026180fd191b6a

#### On Raspberry Pi 1 to 3 (10.0.0.1- 10.0.0.1)
1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Romyull-Islam/hsmdif-llama.git
   cd hsmdif-llama
   ```

2. **Clone `llama.cpp`**:

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   ```

3. **Install Dependencies**:

   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake libboost-all-dev nlohmann-json3-dev libgrpc++-dev protobuf-compiler-grpc grpc-tools python3-pip
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

   - **Note**: If your model is already in GGUF format, skip the conversion step.
   - Ensure you have the necessary Python dependencies for `convert_hf_to_gguf.py`:
     ```bash
     pip install torch transformers
     ```

2. **Copy Models to All Devices**:

   Use `scp` to copy the GGUF files to each Raspberry Pi:

   ```bash
   scp models/llama_30b_q4_0.gguf user@10.0.0.1:~/hsmdif-llama/models/
   scp models/llama_30b_q8_0.gguf user@10.0.0.1:~/hsmdif-llama/models/
   scp models/llama_30b_q4_0.gguf user@10.0.0.2:~/hsmdif-llama/models/
   scp models/llama_30b_q8_0.gguf user@10.0.0.2:~/hsmdif-llama/models/
   scp models/llama_30b_q4_0.gguf user@10.0.0.3:~/hsmdif-llama/models/
   scp models/llama_30b_q8_0.gguf user@10.0.0.3:~/hsmdif-llama/models/
   ```

   Replace `user` with the actual username on the Raspberry Pis.

#### Create `devices.json` (on Mac Mini)
Create a `devices.json` file in the `hsmdif-llama` directory with your cluster configuration:

```json
{
  "devices": [
    {
      "address": "192.168.1.10",
      "port": 9999,
      "total_memory": 16000,
      "memfree": 14000,
      "speed": 10.0,
      "load": 0.1
    },
    {
      "address": "10.0.0.1",
      "port": 9999,
      "total_memory": 16000,
      "memfree": 14000,
      "speed": 5.0,
      "load": 0.2
    },
    {
      "address": "10.0.0.2",
      "port": 9999,
      "total_memory": 8000,
      "memfree": 7000,
      "speed": 4.0,
      "load": 0.3
    },
    {
      "address": "10.0.0.3",
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

#### On Raspberry Pi 1 (10.0.0.1)
```bash
cd hsmdif-llama
./build/hsmdif --worker --model models/llama_30b_q4_0.gguf --port 9999 --nthreads 4
```

**Expected Output**:
```
Worker server listening on 0.0.0.0:9999
```

#### On Raspberry Pi 2 (10.0.0.2)
```bash
cd hsmdif-llama
./build/hsmdif --worker --model models/llama_30b_q4_0.gguf --port 9999 --nthreads 4
```

**Expected Output**:
```
Worker server listening on 0.0.0.0:9999
```

#### On Raspberry Pi 3 (10.0.0.3)
```bash
cd hsmdif-llama
./build/hsmdif --worker --model models/llama_30b_q4_0.gguf --port 9999 --nthreads 4
```

**Expected Output**:
```
Worker server listening on 0.0.0.0:9999
```

**Note**: The model weights are loaded on each worker and remain loaded as long as the gRPC server runs. To stop a worker, press `Ctrl+C`, which will unload the weights.

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
<<<<<<< HEAD
Started memory manager on 192.168.1.10
=======
Started memory manager on 192.168.1.100
>>>>>>> ba728aaf3bf058ba4686e33f1e026180fd191b6a
Started memory manager on 10.0.0.1
Started memory manager on 10.0.0.2
Started memory manager on 10.0.0.3
Running inference...
Root node set to 192.168.1.10:9999
Model memory: 15000MB, Context memory: 2000MB, Total: 17000MB
<<<<<<< HEAD
Device 192.168.1.10:9999 allocated 85 tokens (Memory: 14000MB, Speed: 10 tokens/sec)
Device 10.0.0.1:9999 allocated 43 tokens (Memory: 14000MB, Speed: 5 tokens/sec)
Running distributed inference across devices
Device 192.168.1.10:9999 generating 85 tokens
Device 10.0.0.1:9999 generating 43 tokens
Output: [Inference output]
Inference speed: 12 tokens/second
Inference completed.
Stopped memory manager on 192.168.1.10
Stopped memory manager on 10.0.0.1
Stopped memory manager on 10.0.0.2
Stopped memory manager on 10.0.0.3
```

- **Note**: In non-interactive mode, the weights are loaded for this single inference and then unloaded when the process exits.

#### Interactive Mode (Multiple Inferences with Persistent Weights)
To keep the weights loaded across multiple inferences, use interactive mode:

```bash
cd hsmdif-llama
python run_hsmdif.py \
  --model models/llama_30b_q4_0.gguf \
  --tokenizer models/llama_30b_q4_0.gguf \
  --config devices.json \
  --prioritize-by-memory \
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
Started memory manager on 192.168.1.10
Started memory manager on 10.0.0.1
Started memory manager on 10.0.0.2
Started memory manager on 10.0.0.3
Running inference...
Root node set to 192.168.1.10:9999
Model memory: 15000MB, Context memory: 2000MB, Total: 17000MB
Entering interactive mode. Enter prompts (or 'exit' to quit):
Prompt: Tell me about distributed inference.
Device 192.168.1.10:9999 allocated 85 tokens (Memory: 14000MB, Speed: 10 tokens/sec)
Device 10.0.0.1:9999 allocated 43 tokens (Memory: 14000MB, Speed: 5 tokens/sec)
Running distributed inference across devices
Device 192.168.1.10:9999 generating 85 tokens
Device 10.0.0.1:9999 generating 43 tokens
Output: [Inference output]
Inference speed: 12 tokens/second

Prompt: What is pipeline parallelism?
Device 192.168.1.10:9999 allocated 85 tokens (Memory: 14000MB, Speed: 10 tokens/sec)
Device 10.0.0.1:9999 allocated 43 tokens (Memory: 14000MB, Speed: 5 tokens/sec)
Running distributed inference across devices
Device 192.168.1.10:9999 generating 85 tokens
Device 10.0.0.1:9999 generating 43 tokens
Output: [Inference output]
Inference speed: 12 tokens/second

Prompt: exit
Interactive mode ended.
Stopped memory manager on 192.168.1.10
Stopped memory manager on 10.0.0.1
Stopped memory manager on 10.0.0.2
Stopped memory manager on 10.0.0.3
```

- **Note**: In interactive mode, the weights are loaded once at the start and remain loaded until you type `exit` or press `Ctrl+C`. This ensures faster subsequent inferences since the model doesn't need to be reloaded.

#### Using Custom Priority
To prioritize Raspberry Pi 1 (10.0.0.1) as the root node:

```bash
cd hsmdif-llama
python run_hsmdif.py \
  --model models/llama_30b_q4_0.gguf \
  --tokenizer models/llama_30b_q4_0.gguf \
  --config devices.json \
  --priority 10.0.0.1 192.168.1.10 10.0.0.2 10.0.0.3 \
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
Started memory manager on 192.168.1.10
Started memory manager on 10.0.0.1
Started memory manager on 10.0.0.2
Started memory manager on 10.0.0.3
Running inference...
Root node set to 10.0.0.1:9999
Model memory: 15000MB, Context memory: 2000MB, Total: 17000MB
Entering interactive mode. Enter prompts (or 'exit' to quit):
Prompt: Tell me about distributed inference.
Device 10.0.0.1:9999 allocated 43 tokens (Memory: 14000MB, Speed: 5 tokens/sec)
Device 192.168.1.10:9999 allocated 85 tokens (Memory: 14000MB, Speed: 10 tokens/sec)
Running distributed inference across devices
Device 10.0.0.1:9999 generating 43 tokens
Device 192.168.1.10:9999 generating 85 tokens
Output: [Inference output]
Inference speed: 12 tokens/second
Prompt: exit
Interactive mode ended.
Stopped memory manager on 192.168.1.10
Stopped memory manager on 10.0.0.1
Stopped memory manager on 10.0.0.2
Stopped memory manager on 10.0.0.3
```

## Troubleshooting

- **GGUF Issues**: If the GGUF file fails to load, test it directly with `llama.cpp`:
  ```bash
  cd hsmdif-llama/llama.cpp
  ./llama-cli -m ../models/llama_30b_q4_0.gguf -p "Hello"
  ```
  - The `--model` and `--tokenizer` both point to the same GGUF file because `llama.cpp` extracts the tokenizer from the GGUF file, ensuring compatibility.
- **SSH Issues**: If SSH connections fail, ensure the credentials in `run_hsmdif.py` and `profile_devices.py` are correct, or use SSH keys as shown above.
- **Port Issues**: Verify that port 9999 is open on all devices:
  ```bash
  netstat -tuln | grep 9999
  ```
- **Memory Issues**: If a device runs out of memory, HSM-DIF will distribute the load across more devices. Ensure `devices.json` reflects accurate memory values.
=======
Device 192.168.1.100:9999 allocated 85 tokens (Memory: 14000MB, Speed: 10 tokens/sec)
``
