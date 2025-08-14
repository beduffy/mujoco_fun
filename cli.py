import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Headless robotics sim suite")
    ap.add_argument("command", choices=["run", "bench", "test"], help="run suite, benchmark, or tests")
    ap.add_argument("--seeds", type=int, default=1, help="seeds for benchmark")
    ap.add_argument("--no-mujoco", action="store_true")
    args = ap.parse_args()

    if args.command == "run":
        cmd = [sys.executable, "run_all.py"]
        if args.no_mujoco:
            cmd.append("--no-mujoco")
        subprocess.check_call(cmd)
        subprocess.check_call([sys.executable, "tools/export_gifs.py"]) 
    elif args.command == "bench":
        subprocess.check_call([sys.executable, "tools/benchmark.py", "--seeds", str(args.seeds)])
    elif args.command == "test":
        subprocess.check_call([sys.executable, "-m", "pytest", "-q"])


if __name__ == "__main__":
    main()