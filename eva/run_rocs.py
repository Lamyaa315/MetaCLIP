import argparse
import json
from pathlib import Path

from eva.loader import load_eva_model
from clipeval.rocs.eval_rocs import evaluate_rocs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["eva02_b16", "eva02_l14"],
    )

    parser.add_argument(
        "--config",
        required=True,
        choices=["coco", "flickr30k"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full dataset",
    )

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"ROCS config: {args.config}")
    print(f"Limit: {args.limit if args.limit else 'full'}")

    model, preprocess, tokenizer, device = load_eva_model(args.model)

    results = evaluate_rocs(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        config=args.config,
        limit=args.limit,
    )

    output_dir = Path("logs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"rocs_{args.config}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults:")
    print(json.dumps(results, indent=4))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
