#!/usr/bin/env python3
import paramiko
import psutil
import json
import argparse
import time
import os

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

def profile_device(host, port):
    # Get memory stats via SSH
    mem_command = "free -m"
    mem_output = ssh_execute_command(host, mem_command)
    mem_lines = mem_output.splitlines()
    if len(mem_lines) < 2:
        raise Exception(f"Failed to parse memory info from {host}")
    mem_data = mem_lines[1].split()
    total_memory = int(mem_data[1])
    memfree = int(mem_data[3])  # Available memory

    # Get CPU load
    load_command = "uptime"
    load_output = ssh_execute_command(host, load_command)
    load_avg = float(load_output.split("load average:")[1].split(",")[0].strip())

    # Estimate speed (tokens per second) based on CPU cores and a benchmark
    speed_command = "nproc"
    cores = int(ssh_execute_command(host, speed_command).strip())
    # Rough estimate: 2 tokens/sec per core (adjusted based on device type)
    speed = cores * 2.0
    # Adjust speed based on device (heuristic)
    if "raspberry" in host.lower() or "pi" in host.lower():
        speed *= 0.5  # Raspberry Pis are slower
    speed = max(1.0, speed)  # Minimum speed of 1 token/sec

    return {
        "address": host,
        "port": port,
        "total_memory": total_memory,
        "memfree": memfree,
        "speed": speed,
        "load": load_avg
    }

def main():
    parser = argparse.ArgumentParser(description="Profile devices for HSM-DIF")
    parser.add_argument("--devices", required=True, help="Comma-separated list of devices (address:port)")
    args = parser.parse_args()

    devices = []
    for device in args.devices.split(","):
        addr, port = device.split(":")
        devices.append((addr, int(port)))

    device_profiles = []
    for host, port in devices:
        print(f"Profiling {host}:{port}...")
        profile = profile_device(host, port)
        device_profiles.append(profile)

    # Output to JSON
    output = {"devices": device_profiles}
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()