import argparse
from src.train import train
from src.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Action Spotting in Soccer Videos")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)

    # Data
    parser.add_argument("--data_dir", default="data/soccernet")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_model.pt")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Model
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Evaluation
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms_window", type=int, default=10)
    parser.add_argument("--output_dir", default="results/predictions")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            window_size=args.window_size,
            stride=args.stride,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.mode == "evaluate":
        run_evaluation(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint_path,
            split=args.split,
            window_size=args.window_size,
            stride=args.stride,
            threshold=args.threshold,
            nms_window=args.nms_window,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
