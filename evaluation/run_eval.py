import argparse
import subprocess
import os
import sys


def run_command(cmd):
    print("\n[RUNNING]")
    print(" ".join(cmd))
    print("-" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: inference + evaluation")

    # Core
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g. PolypPVT)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test datasets root')

    # Optional
    parser.add_argument('--testset_names', nargs='+', default=None,
                        help='Dataset names (optional)')
    parser.add_argument('--testsize', type=int, default=352)

    parser.add_argument('--pth_path', type=str, default=None)
    parser.add_argument('--pred_base', type=str, default=None)
    parser.add_argument('--gt_base', type=str, default=None)

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_area', type=int, default=0)

    args = parser.parse_args()

    # -------------------------
    # Resolve defaults
    # -------------------------
    if args.pth_path is None:
        args.pth_path = os.path.join("models", args.model, "best.pth")

    if args.pred_base is None:
        args.pred_base = os.path.join("outputs", "predictions", args.model)

    if args.gt_base is None:
        args.gt_base = args.data_path

    # -------------------------
    # 1. Run Inference
    # -------------------------
    inference_script = os.path.join("inference", "generate_predictions.py")

    cmd_infer = [
        sys.executable, inference_script,
        "--model", args.model,
        "--data_path", args.data_path,
        "--testsize", str(args.testsize),
        "--pth_path", args.pth_path,
        "--save_path", args.pred_base
    ]

    if args.testset_names is not None:
        cmd_infer += ["--testset_names"] + args.testset_names

    run_command(cmd_infer)

    # -------------------------
    # 2. Run Evaluation
    # -------------------------
    eval_script = os.path.join("evaluation", "evaluate_predictions.py")

    cmd_eval = [
        sys.executable, eval_script,
        "--pred_base", args.pred_base,
        "--gt_base", args.gt_base,
        "--threshold", str(args.threshold),
        "--min_area", str(args.min_area)
    ]

    if args.testset_names is not None:
        cmd_eval += ["--datasets"] + args.testset_names

    run_command(cmd_eval)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()