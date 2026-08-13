import torch
import open_clip

from eva.configs import EVA_MODELS


def load_eva_model(config_name):
    cfg = EVA_MODELS[config_name]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model_name"],
        pretrained=cfg["pretrained"],
    )

    tokenizer = open_clip.get_tokenizer(
        cfg["model_name"]
    )

    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device
