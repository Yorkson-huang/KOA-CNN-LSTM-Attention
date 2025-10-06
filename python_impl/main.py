"""Entry point that mirrors MAIN.m and optionally runs KOA optimization."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .koa import koa
    from .objective_function import objective_function
except ImportError:  # Allow running the file as a script
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from python_impl.koa import koa  # type: ignore
    from python_impl.objective_function import objective_function  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KOA-CNN-LSTM-Attention Python port")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the Excel dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device identifier")

    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--kernel-size", type=float, default=3.0, help="Convolution kernel size")
    parser.add_argument("--num-neurons", type=float, default=32.0, help="LSTM hidden units")

    parser.add_argument(
        "--use-koa",
        action="store_true",
        help="Run the KOA optimizer instead of a single training run.",
    )
    parser.add_argument("--agents", type=int, default=5, help="Number of KOA agents")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum KOA iterations")
    parser.add_argument(
        "--lower-bounds",
        type=float,
        nargs=3,
        default=(0.001, 1, 8),
        help="Lower bounds for [learning_rate, kernel_size, num_neurons]",
    )
    parser.add_argument(
        "--upper-bounds",
        type=float,
        nargs=3,
        default=(0.1, 7, 128),
        help="Upper bounds for [learning_rate, kernel_size, num_neurons]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.use_koa:
        result = koa(
            search_agents=args.agents,
            t_max=args.iterations,
            upper_bounds=args.upper_bounds,
            lower_bounds=args.lower_bounds,
            dim=3,
            data_path=args.data_path,
            device=device,
        )
        print("KOA best score (MAPE surrogate):", result.best_score)
        print("KOA best position [lr, kernel, neurons]:", result.best_position)
        print("Sorted final fitness values:", result.curve)
    else:
        params = [args.learning_rate, args.kernel_size, args.num_neurons]
        outcome = objective_function(params, data_path=args.data_path, device=device)
        print("MAPE surrogate:", outcome.loss_metric)
        print("Metrics:", outcome.metrics)


if __name__ == "__main__":
    main()

