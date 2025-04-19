#!/usr/bin/env python3
import psutil
import time
import os

def get_inactive_memory():
    # On Linux/macOS, use psutil to get memory stats
    mem = psutil.virtual_memory()
    # psutil on macOS uses 'inactive' memory; on Linux, it's approximated
    inactive = getattr(mem, 'inactive', mem.cached + mem.buffers)
    return inactive / (1024 * 1024)  # Convert to MB

def reclaim_inactive_memory():
    # On Linux, use 'sync; sysctl vm.drop_caches=3' (requires root)
    # On macOS, use 'purge' command (requires root)
    if os.uname().sysname == "Linux":
        try:
            os.system("sudo sync; sudo sysctl -w vm.drop_caches=3")
        except:
            pass  # Silently fail if not root
    elif os.uname().sysname == "Darwin":
        try:
            os.system("sudo purge")
        except:
            pass  # Silently fail if not root

def main():
    print("Starting inactive memory manager...")
    while True:
        inactive_mb = get_inactive_memory()
        print(f"Inactive memory: {inactive_mb:.2f} MB")
        if inactive_mb > 500:  # Threshold: 500MB
            print("Reclaiming inactive memory...")
            reclaim_inactive_memory()
        time.sleep(60)  # Check every 60 seconds

if __name__ == "__main__":
    main()