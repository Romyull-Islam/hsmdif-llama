#!/usr/bin/env python3
import subprocess
import argparse
import paramiko
import time
import os
import signal
import sys

# SSH credentials (replace with your actual credentials or use SSH keys)
SSH_USERNAME = "user"
SSH_PASSWORD = "password"

def ssh_execute_command(host, command, username=SSH_USERNAME, password=SSH_PASSWORD):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=username, password=password, timeout=10)
        stdin, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        if exit_status != 0:
            raise Exception(f"Command failed on {host}: {error}")
        return output
    finally:
        client.close()

def start_memory_manager(host, project_dir):
    command = f"cd {project_dir} && python3 scripts/inactive_memory_manager.py &"
    try:
        ssh_execute_command(host, command)
        print(f"Started memory manager on {host}")
    except Exception as e:
        print(f"Failed to start memory manager on {host}: {e}")

def stop_memory_manager(host, project_dir):
    command = f"pkill -f 'python3 scripts/inactive_memory_manager.py'"
    try:
        ssh_execute_command(host, command)
        print(f"Stopped memory manager on {host}")
    except Exception as e:
        print(f"Failed to stop memory manager on {host}: {e}")

def profile_devices(devices):
    devices_str = ",".join([f"{addr}:{port}" for addr, port in devices])
    command = f"python3 scripts/profile_devices.py --devices \"{devices_str}\" > temp_devices.json"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to profile devices: {result.stderr}")
    print("Profiling devices...")

def update_device_metrics(devices, project_dir):
    print("Updating device metrics...")
    for host, _ in devices:
        command = f"cd {project_dir} && python3 scripts/profile_devices.py --devices \"{host}:9999\" > temp.json"
        try:
            ssh_execute_command(host, command)
        except Exception as e:
            print(f"Failed to update metrics for {host}: {e}")

def run_inference(args, devices, project_dir):
    devices_str = ",".join([f"{addr}:{port}" for addr, port in devices])
    priority_str = ",".join(args.priority) if args.priority else ""

    # Base command for hsmdif
    command = [
        "./build/hsmdif",
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--workers", devices_str,
        "--prompt", "",  # Placeholder for the prompt
        "--max-seq-len", str(args.max_seq_len),
        "--max-tokens", str(args.max_tokens),
        "--nthreads", str(args.nthreads),
        "--quantize", args.quantize,
        "--num-parameters", str(args.num_parameters)
    ]

    if priority_str:
        command.extend(["--priority", priority_str])
    if args.prioritize_by_memory:
        command.append("--prioritize-by-memory")

    # Start the hsmdif process
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        process.terminate()
        process.wait()
        # Stop memory managers on all devices
        for host, _ in devices:
            stop_memory_manager(host, project_dir)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if args.non_interactive:
        # Single inference
        process.stdin.write(args.prompt + "\n")
        process.stdin.flush()
        output, error = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Inference failed: {error}")
        print("Inference completed.")
        print(output)
        # Stop memory managers
        for host, _ in devices:
            stop_memory_manager(host, project_dir)
    else:
        # Interactive mode: keep weights loaded and accept multiple prompts
        print("Entering interactive mode. Enter prompts (or 'exit' to quit):")
        while True:
            prompt = input("Prompt: ")
            if prompt.lower() == "exit":
                break
            process.stdin.write(prompt + "\n")
            process.stdin.flush()
            
            # Read output until we get the inference speed (end of inference)
            output = ""
            while True:
                line = process.stdout.readline()
                output += line
                if "Inference speed" in line:
                    break
            print(output)
        
        # Terminate the process
        process.terminate()
        process.wait()
        print("Interactive mode ended.")
        # Stop memory managers
        for host, _ in devices:
            stop_memory_manager(host, project_dir)

def main():
    parser = argparse.ArgumentParser(description="HSM-DIF: Hybrid Speed-Memory-Aware Distributed Inference Framework")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode (single inference)")
    parser.add_argument("--model", required=True, help="Path to the model file (GGUF format)")
    parser.add_argument("--tokenizer", required=True, help="Path to the tokenizer file (same as model for GGUF)")
    parser.add_argument("--workers", help="Comma-separated list of worker addresses (address:port)")
    parser.add_argument("--config", help="Path to JSON config file for devices")
    parser.add_argument("--priority", nargs="+", help="List of device addresses in priority order")
    parser.add_argument("--prioritize-by-memory", action="store_true", help="Prioritize devices by memory capacity")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt for inference")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length (in tokens)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of threads")
    parser.add_argument("--quantize", default="8bit", choices=["4bit", "8bit"], help="Quantization level")
    parser.add_argument("--num-parameters", type=int, default=3000000000, help="Number of model parameters")

    args = parser.parse_args()

    # Validate arguments
    if not (args.workers or args.config):
        raise ValueError("Must specify either --workers or --config")
    if args.workers and args.config:
        raise ValueError("Cannot specify both --workers and --config")

    # Project directory
    project_dir = os.path.abspath(os.path.dirname(__file__))

    # Parse devices
    devices = []
    if args.config:
        # Devices will be loaded from the config file by hsmdif.cpp
        devices.extend([
            ("192.168.1.100", 9999),  # Mac Mini
            ("10.0.0.179", 9999),     # RPi 1
            ("10.0.0.124", 9999),     # RPi 2
            ("10.0.0.244", 9999)      # RPi 3
        ])
    else:
        for worker in args.workers.split(","):
            addr, port = worker.split(":")
            devices.append((addr, int(port)))

    # Remove the root node (highest-priority device) from the list of devices where memory managers are started
    # Since hsmdif.cpp will set the root node, we'll assume all devices need memory managers for simplicity
    # In practice, you might exclude the root node if running locally

    # Profile devices
    if not args.config:
        profile_devices(devices)

    # Update device metrics
    update_device_metrics(devices, project_dir)

    # Start memory managers on all devices
    print("Starting inactive memory manager on each device...")
    for host, _ in devices:
        start_memory_manager(host, project_dir)
        time.sleep(1)  # Small delay to ensure managers start

    # Run inference
    print("Running inference...")
    run_inference(args, devices, project_dir)

if __name__ == "__main__":
    main()