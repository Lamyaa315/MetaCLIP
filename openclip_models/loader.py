import torch
import open_clip

from openclip_models.configs import OPENCLIP_MODELS


def load_openclip_model(model_key):
    if model_key not in OPENCLIP_MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available: {list(OPENCLIP_MODELS.keys())}"
        )

    cfg = OPENCLIP_MODELS[model_key]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading: {cfg['display_name']}")
    print(f"OpenCLIP model: {cfg['model']}")
    print(f"Checkpoint: {cfg['pretrained']}")
    print(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model"],
        pretrained=cfg["pretrained"],
    )

    tokenizer = open_clip.get_tokenizer(cfg["model"])

    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device
