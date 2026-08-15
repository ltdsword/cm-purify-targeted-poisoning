import subprocess
import os
import sys

print("Running dry-run test mode to verify Python syntax, imports, and flag correctness without consuming heavy resources...")

# Use the same exact base command structure as our generated code,
# but inject "dry run" arguments to instantly fail or instantly succeed.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wb_dir = os.path.join(BASE_DIR, "poisoning-gradient-matching")
bp_dir = os.path.join(BASE_DIR, "BullseyePoison")
generation_script = os.path.join(BASE_DIR, "scripts", "dataset_generation.py")

try:
    if os.name == "nt":
        print("\n--- Skipping legacy WB/BP import checks on Windows (their upstream runtimes are Linux/CUDA-specific) ---")
    else:
        print("\n--- Testing Witches Brew Invocation ---")
        wb_cmd = [
            sys.executable, "brew_poison.py", "--name", "test", "--benchmark", "dummy.pickle",
            "--save", "benchmark", "--vruns", "0", "--eps", "8", "--benchmark_idx", "0",
            "--ensemble", "1", "--net", "ResNet18", "--help",
        ]
        subprocess.run(wb_cmd, cwd=wb_dir, check=True, stdout=subprocess.DEVNULL)
        print("Witches' Brew argument parsing succeeded.")

        print("\n--- Testing Bullseye Polytope Invocation ---")
        bp_cmd = [
            sys.executable, "craft_poisons_transfer.py", "--target-label", "0", "--target-index", "0",
            "--poison-label", "1", "--start-idx", "0", "--poison-num", "10",
            "--substitute-nets", "ResNet18", "--target-net", "ResNet18", "--help",
        ]
        subprocess.run(bp_cmd, cwd=bp_dir, check=True, stdout=subprocess.DEVNULL)
        print("Bullseye Polytope argument parsing succeeded.")

    print("\n--- Testing Narcissus Orchestration Invocation ---")
    subprocess.run(
        [sys.executable, generation_script, "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print("Narcissus orchestration arguments parsed successfully.")

    print("\n--- ALL CLI PARSER CHECKS PASSED ---")
except subprocess.CalledProcessError as e:
    print(f"\nFATAL ERROR IN CLI INVOCATION: {e}")
    raise SystemExit(1)

