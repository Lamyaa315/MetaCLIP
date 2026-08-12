# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset


ROCS = "AbdulmalekDS/ROCS"

ALPHA = 0.4
K = 10


def fixed_crops(im, r=0.6):
    W, H = im.size
    s = int(r * min(W, H))
    cx, cy = W // 2, H // 2

    boxes = [
        (cx - s // 2, cy - s // 2),
        (0, 0),
        (W - s, 0),
        (0, H - s),
        (W - s, H - s),
    ]

    return [
        im.crop((x, y, x + s, y + s))
        for x, y in boxes
    ]


def csls(S, k=K):
    rt = np.sort(S, axis=1)[:, -k:].mean(axis=1)
    ri = np.sort(S, axis=0)[-k:, :].mean(axis=0)
    return 2 * S - rt[:, None] - ri[None, :]


def recall(S, gt):
    ranking = (-S).argsort(axis=1)

    return [
        round(
            100
            * np.mean([
                gt[q] in ranking[q, :k]
                for q in range(len(gt))
            ]),
            2,
        )
        for k in (1, 5, 10)
    ]


def encode_images(model, preprocess, images, device, batch_size=32):
    features = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            batch = torch.stack([
                preprocess(image)
                for image in batch_images
            ]).to(device)

            embeddings = model.encode_image(batch)
            embeddings = F.normalize(embeddings, dim=-1)

            features.append(
                embeddings.cpu().numpy()
            )

    return np.concatenate(features, axis=0)


def encode_texts(model, tokenizer, texts, device, batch_size=128):
    features = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            tokens = tokenizer(batch_texts).to(device)

            embeddings = model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)

            features.append(
                embeddings.cpu().numpy()
            )

    return np.concatenate(features, axis=0)


def evaluate_rocs(
    model,
    preprocess,
    tokenizer,
    device,
    config="coco",
    limit=0,
):
    print(f"Loading ROCS/{config} with streaming=True...")

    dataset = load_dataset(
        ROCS,
        config,
        split="test",
        streaming=True,
    )

    images = []
    captions = []

    for sample in dataset:
        image = sample["image"].convert("RGB")

        image_index = len(images)
        images.append(image)

        for caption in sample["captions"]:
            captions.append((caption, image_index))

        if limit and len(images) >= limit:
            break

    print(f"Loaded {len(images)} images.")
    print(f"Loaded {len(captions)} captions.")

    if not images:
        raise RuntimeError("ROCS dataset returned zero images.")

    text_strings = [caption for caption, _ in captions]

    gt = np.array([
        image_index
        for _, image_index in captions
    ])

    print("Encoding text...")

    text_features = encode_texts(
        model,
        tokenizer,
        text_strings,
        device,
    )

    print("Encoding images...")

    image_features = encode_images(
        model,
        preprocess,
        images,
        device,
    )

    similarity = text_features @ image_features.T

    baseline = recall(similarity, gt)

    print("Baseline R@1/5/10:", baseline)

    print("Computing MINER crops...")

    crop_features = []

    for image in images:
        crops = fixed_crops(image)

        features = encode_images(
            model,
            preprocess,
            crops,
            device,
        )

        crop_features.append(features)

    crop_features = np.stack(crop_features, axis=0)

    crop_similarity = np.einsum(
        "qd,nkd->qnk",
        text_features,
        crop_features,
    )

    miner_similarity = np.max(
        crop_similarity,
        axis=2,
    )

    miner_similarity = (
        (1 - ALPHA) * similarity
        + ALPHA * miner_similarity
    )

    miner_similarity = csls(miner_similarity)

    miner = recall(miner_similarity, gt)

    print("MINER R@1/5/10:", miner)

    return {
        "baseline": baseline,
        "miner": miner,
        "n_images": len(images),
    }


def main(
    model,
    preprocess_val,
    tokenizer,
    result_json,
    device,
):
    config = os.environ.get("ROCS_CONFIG", "coco")
    limit = int(os.environ.get("ROCS_LIMIT", "0"))

    results = evaluate_rocs(
        model=model,
        preprocess=preprocess_val,
        tokenizer=tokenizer,
        device=device,
        config=config,
        limit=limit,
    )

    with open(result_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved ROCS results to {result_json}")


def parse_results(results, result_json):
    with open(result_json) as f:
        rocs_result = json.load(f)

    print("ROCS baseline:", rocs_result["baseline"])
    print("ROCS MINER:", rocs_result["miner"])

    results["rocs_baseline_R1"] = rocs_result["baseline"][0]
    results["rocs_baseline_R5"] = rocs_result["baseline"][1]
    results["rocs_baseline_R10"] = rocs_result["baseline"][2]

    results["rocs_miner_R1"] = rocs_result["miner"][0]
    results["rocs_miner_R5"] = rocs_result["miner"][1]
    results["rocs_miner_R10"] = rocs_result["miner"][2]
