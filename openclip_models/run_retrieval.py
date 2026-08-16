import argparse
import csv
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from openclip_models.loader import load_openclip_model
from openclip_models.configs import OPENCLIP_MODELS


DATASETS = {
    "coco": {
        "hf_name": "undefined443/coco-karpathy-wds",
        "split": "test",
        "display_name": "COCO",
    },
    "flickr30k": {
        "hf_name": "imirandam/flickr30k_karpathy_test_split",
        "split": "test",
        "display_name": "Flickr30K",
    },
}

MODEL_NAMES = {
    key: cfg["display_name"]
    for key, cfg in OPENCLIP_MODELS.items()
}


def load_retrieval_dataset(name, limit=0):
    cfg = DATASETS[name]

    print(f"Loading original {cfg['display_name']} with streaming=True...")

    ds = load_dataset(
        cfg["hf_name"],
        split=cfg["split"],
        streaming=True,
    )

    images = []
    captions = []
    image_to_caption_indices = []

    for i, sample in enumerate(ds):
        if limit and i >= limit:
            break

        if name == "coco":
            image = sample["jpg"].convert("RGB")
            caps = sample["json"]["captions"]
        else:
            image = sample["filename"].convert("RGB")
            caps = sample["caption"]

        start = len(captions)

        images.append(image)
        captions.extend(caps)
        image_to_caption_indices.append(
            list(range(start, start + len(caps)))
        )

    print(f"Loaded {len(images)} images.")
    print(f"Loaded {len(captions)} captions.")

    return images, captions, image_to_caption_indices


@torch.inference_mode()
def encode_texts(model, tokenizer, texts, device, batch_size=256):
    features = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch = texts[i:i + batch_size]

        tokens = tokenizer(batch).to(device)

        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        features.append(feats.cpu())

    return torch.cat(features, dim=0)


@torch.inference_mode()
def encode_images(model, preprocess, images, device, batch_size=64):
    features = []

    for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
        batch_images = images[i:i + batch_size]

        batch = torch.stack(
            [preprocess(image) for image in batch_images]
        ).to(device)

        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        features.append(feats.cpu())

    return torch.cat(features, dim=0)


def compute_image_to_text_recall(
    image_features,
    text_features,
    image_to_caption_indices,
):
    similarities = image_features @ text_features.T

    recalls = {}

    for k in [1, 5, 10]:
        topk = similarities.topk(k, dim=1).indices

        correct = 0

        for image_idx, valid_caption_indices in enumerate(
            image_to_caption_indices
        ):
            retrieved = set(topk[image_idx].tolist())
            valid = set(valid_caption_indices)

            if retrieved & valid:
                correct += 1

        recalls[k] = round(
            100.0 * correct / len(image_to_caption_indices),
            2
        )

    return recalls


def update_csv(model_name, dataset_name, recalls, n_images):
    csv_path = Path("logs/retrieval_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Model",
        "Dataset",
        "Method",
        "R1",
        "R5",
        "R10",
        "N_Images",
    ]

    existing_rows = []

    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if not (
                    row[0] == model_name
                    and row[1] == dataset_name
                    and row[2] == "Baseline"
                ):
                    existing_rows.append(row)

    new_row = [
        model_name,
        dataset_name,
        "Baseline",
        recalls[1],
        recalls[5],
        recalls[10],
        n_images,
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows)
        writer.writerow(new_row)

    print(f"Updated results in: {csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(OPENCLIP_MODELS.keys()),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["coco", "flickr30k"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full test split",
    )

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Dataset: {DATASETS[args.dataset]['display_name']}")
    print(f"Limit: {args.limit if args.limit else 'full'}")

    model, preprocess, tokenizer, device = load_openclip_model(args.model)

    images, captions, mappings = load_retrieval_dataset(
        args.dataset,
        args.limit,
    )

    text_features = encode_texts(
        model,
        tokenizer,
        captions,
        device,
    )

    image_features = encode_images(
        model,
        preprocess,
        images,
        device,
    )

    recalls = compute_image_to_text_recall(
        image_features,
        text_features,
        mappings,
    )

    print(
        f"\nImage -> Text R@1/5/10: "
        f"[{recalls[1]}, {recalls[5]}, {recalls[10]}]"
    )

    output_dir = Path("logs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.dataset}.json"

    result = {
        "model": MODEL_NAMES[args.model],
        "dataset": DATASETS[args.dataset]["display_name"],
        "task": "image_to_text_retrieval",
        "R1": recalls[1],
        "R5": recalls[5],
        "R10": recalls[10],
        "n_images": len(images),
        "n_captions": len(captions),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    update_csv(
        MODEL_NAMES[args.model],
        DATASETS[args.dataset]["display_name"],
        recalls,
        len(images),
    )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
