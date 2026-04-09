import argparse
import json
import random
from typing import Dict, List, Tuple

import torch
import yaml
from tqdm import tqdm

from models.multimodal_model import create_multimodal_model
from data.caption_dataset import build_val_dataloaders
from utils.checkpoint import create_checkpoint_manager


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

@torch.no_grad()
def collect_embeddings(model, dataloader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    收集整个评估集上的 image/text embeddings

    Returns:
        image_embeds: [N, D]
        text_embeds: [N, D]
    """
    model.eval()

    all_image_embeds: List[torch.Tensor] = []
    all_text_embeds: List[torch.Tensor] = []

    pbar = tqdm(dataloader, desc="Collect embeddings", dynamic_ncols=True)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = outputs["image_embeds"]
        text_embeds = outputs["text_embeds"]

        if image_embeds.shape[-1] != text_embeds.shape[-1]:
            raise ValueError(
                f"image_embeds dim ({image_embeds.shape[-1]}) != "
                f"text_embeds dim ({text_embeds.shape[-1]})"
            )

        image_embeds = l2_normalize(image_embeds)
        text_embeds = l2_normalize(text_embeds)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [N, D]

    return all_image_embeds, all_text_embeds

def compute_recall_at_k(similarity: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """
    计算一一对应场景下的 Recall@K

    假设第 i 个 image 对应第 i 个 text
    similarity: [N, N]
    """
    num_samples = similarity.size(0)
    gt = torch.arange(num_samples)

    results: Dict[str, float] = {}

    max_k = min(max(ks), num_samples)

    # Image -> Text
    i2t_topk = similarity.topk(k=max_k, dim=1).indices  # [N, max_k]
    for k in ks:
        correct = (i2t_topk[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"I2T_R@{k}"] = correct

    # Text -> Image
    t2i_topk = similarity.t().topk(k=max_k, dim=1).indices  # [N, max_k]
    for k in ks:
        correct = (t2i_topk[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"T2I_R@{k}"] = correct

    return results

def print_metrics(metrics: Dict[str, float], title: str = "Recall@K") -> None:
    print("=" * 80)
    print(title)
    print("-" * 80)
    for key in sorted(metrics.keys()):
        print(f"{key:<12}: {metrics[key] * 100:.2f}%")
    print("=" * 80)

def save_metrics(metrics: Dict[str, float], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {save_path}")


def load_model_from_checkpoint(config: dict, checkpoint_path: str, device: torch.device):
    """
    从 checkpoint 加载训练后的模型
    """
    model = create_multimodal_model(config)
    checkpoint_manager = create_checkpoint_manager(config["output_dir"])
    checkpoint_manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
    )
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained multimodal model Recall@K")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_stage1.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint, e.g. checkpoint-best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override num_workers",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save metrics json",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = args.batch_size if args.batch_size is not None else config["dataset"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["dataset"]["num_workers"]

    # 构建数据
    val_dataset, eval_loader = build_val_dataloaders(
        vision_model_name=config["model"]["vision_encoder"]["model_name"],
        qwen_model_name=config["model"]["llm"]["model_name"],
        split=args.split,
        train_ratio=config["dataset"]["train_ratio"],
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=config["dataset"]["max_length"],
        sample_one_caption=True,  # 改为True，与训练一致
        add_prompt=False,
        seed=config["seed"],
    )

    # 加载训练后模型
    model = load_model_from_checkpoint(config, args.checkpoint, device)

    # 收集 embeddings
    image_embeds, text_embeds = collect_embeddings(model, eval_loader, device)
    print(f"Collected {image_embeds.size(0)} image-text pairs for evaluation.")
    # 相似度矩阵
    text_embeds = text_embeds.to(image_embeds.dtype)
    similarity = image_embeds @ text_embeds.t()  # [N, N]
    # 应用 logit_scale 如果模型有的话
    if hasattr(model, 'logit_scale'):
        similarity = model.logit_scale.item() * similarity + model.logit_bias.item()

    # Recall@K
    metrics = compute_recall_at_k(similarity, ks=(1, 5, 10))

    print_metrics(metrics, title="Trained Model Recall@K")

    if args.save_json is not None:
        save_metrics(metrics, args.save_json)


if __name__ == "__main__":
    main()