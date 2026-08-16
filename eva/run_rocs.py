import argparse
import csv
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
	    # Append results to the central CSV file
    csv_path = Path("logs/retrieval_results.csv")

    model_names = {
        "eva02_b16": "EVA02-B-16",
        "eva02_l14": "EVA02-L-14",
    }

    dataset_names = {
        "coco": "ROCS-COCO",
        "flickr30k": "ROCS-Flickr30K",
    }

    rows = [
        [
            model_names[args.model],
            dataset_names[args.config],
            "Baseline",
            results["baseline"][0],
            results["baseline"][1],
            results["baseline"][2],
            results["n_images"],
        ],
        [
            model_names[args.model],
            dataset_names[args.config],
            "MINER",
            results["miner"][0],
            results["miner"][1],
            results["miner"][2],
            results["n_images"],
        ],
    ]

        # Read existing results and remove duplicates for this model/dataset
    existing_rows = []

    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for row in reader:
                if not (
                    row[0] == model_names[args.model]
                    and row[1] == dataset_names[args.config]
                ):
                    existing_rows.append(row)

    # Rewrite CSV with old results + latest results
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["Model", "Dataset", "Method", "R1", "R5", "R10", "N_Images"]
        )

        writer.writerows(existing_rows)
        writer.writerows(rows)

    print(f"Updated results in: {csv_path}")
    print("\nResults:")
    print(json.dumps(results, indent=4))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
