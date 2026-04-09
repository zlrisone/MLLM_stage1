import torch
from torch.utils.data import DataLoader,random_split
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

import random
from typing import Dict, List, Optional,Any


class CocoCaptionAlignmentCollator:
    """
    将 lmms-lab/COCO-Caption 的样本整理成:
        pixel_values: Tensor [B, 3, H, W]
        input_ids: Tensor [B, T]
        attention_mask: Tensor [B, T]

    图像:
        processor(images=[...], return_tensors="pt")

    文本:
        tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    """
    def __init__(
        self,
        vision_model_name: str,
        qwen_model_name: str,
        max_length: int = 64,
        padding: str = "longest",
        truncation: bool = True,
        sample_one_caption: bool = True,
        add_prompt: bool = False,
    ):
        self.processor = AutoProcessor.from_pretrained(vision_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
        )

        # Qwen 系 tokenizer 有时没有 pad_token，训练时最好补齐
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.sample_one_caption = sample_one_caption
        self.add_prompt = add_prompt
    def _select_caption(self, example: Dict) -> str:
        """
        从 example["answer"] 中取一条 caption。
        COCO-Caption 的 answer 是一个字符串列表。
        """
        answers = example["answer"]

        if isinstance(answers, list):
            answers = [x for x in answers if isinstance(x, str) and len(x.strip()) > 0]
            if len(answers) == 0:
                caption = ""
            elif self.sample_one_caption:
                caption = random.choice(answers)
            else:
                caption = answers[0]
        elif isinstance(answers, str):
            caption = answers
        else:
            caption = ""

        caption = caption.strip()

        if self.add_prompt:
            caption = f"Please carefully observe the image and come up with a caption for the image. {caption}"

        return caption
        
    def __call__(self, batch: List[Dict]) ->  Dict[str, Any]:
        images = []
        texts = []

        for example in batch:
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            caption = self._select_caption(example)

            images.append(image)
            texts.append(caption)

        # vision
        vision_inputs = self.processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = vision_inputs["pixel_values"]  # [B, 3, H, W]

        # text
        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
        )

        input_ids = text_inputs["input_ids"]               # [B, T]
        attention_mask = text_inputs["attention_mask"]     # [B, T]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_texts": texts,  # 调试可用，不需要可删除
        }

def build_coco_caption_dataloader(
    vision_model_name: str,
    qwen_model_name: str,
    split: str = "val",
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 64,
    shuffle: bool = True,
    sample_one_caption: bool = True,
    add_prompt: bool = False,
    streaming: bool = False,
):
    """
    构建 lmms-lab/COCO-Caption 的 DataLoader

    Args:
        vision_model_name: 视觉 processor 对应名称
        qwen_model_name: 文本 tokenizer 对应名称
        split: "val" 或 "test"
        batch_size: batch size
        num_workers: dataloader workers
        max_length: 文本最大长度
        shuffle: 是否打乱
        sample_one_caption: answer 为多 caption 时，是否随机采样一条
        add_prompt: 是否把 question 拼到 caption 前
        streaming: 是否启用 streaming

    Returns:
        dataset, dataloader
    """
    dataset = load_dataset(
        "lmms-lab/COCO-Caption",
        split=split,
        streaming=streaming,
        cache_dir="./hf_cache"
    )

    collator = CocoCaptionAlignmentCollator(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        max_length=max_length,
        sample_one_caption=sample_one_caption,
        add_prompt=add_prompt,
    )

    # streaming dataset 和普通 dataset 在 DataLoader 参数上略有差异
    if streaming:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    return dataset, dataloader

def build_train_val_dataloaders(
    vision_model_name: str,
    qwen_model_name: str,
    split: str = "val",
    train_ratio: float = 0.9,
    batch_size: int = 8,
    num_workers: int = 0,
    max_length: int = 64,
    sample_one_caption: bool = True,
    add_prompt: bool = False,
    seed: int = 42,
):
    dataset, _ = build_coco_caption_dataloader(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        max_length=max_length,
        shuffle=False,
        sample_one_caption=sample_one_caption,
        add_prompt=add_prompt,
        streaming=False,
    )

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 重新构造 collator
    
    collator = CocoCaptionAlignmentCollator(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        max_length=max_length,
        sample_one_caption=sample_one_caption,
        add_prompt=add_prompt,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collator,
    )

    return train_dataset, val_dataset, train_loader, val_loader
def build_val_dataloaders(
    vision_model_name: str,
    qwen_model_name: str,
    split: str = "val",
    train_ratio: float = 0.9,
    batch_size: int = 8,
    num_workers: int = 0,
    max_length: int = 64,
    sample_one_caption: bool = True,
    add_prompt: bool = False,
    seed: int = 42,
):
    dataset, _ = build_coco_caption_dataloader(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        max_length=max_length,
        shuffle=False,
        sample_one_caption=sample_one_caption,
        add_prompt=add_prompt,
        streaming=False,
    )

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 重新构造 collator
    
    collator = CocoCaptionAlignmentCollator(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        max_length=max_length,
        sample_one_caption=sample_one_caption,
        add_prompt=add_prompt,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collator,
    )

    return val_dataset, val_loader
if __name__=="__main__":
    vision_model_name = "google/siglip2-base-patch16-224"
    qwen_model_name = "Qwen/Qwen2.5-3B"

    dataset, dataloader = build_coco_caption_dataloader(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        split="val",
        batch_size=8,
        max_length=64,
        # shuffle=True,
        sample_one_caption=True,
        add_prompt=False,
        streaming = True,
    )

    batch = next(iter(dataloader))

    print(batch["pixel_values"].shape)    # [B, 3, H, W] [8, 3, 224, 224]
    print(batch["input_ids"].shape)       # [B, T] [8, 19]
    print(batch["attention_mask"].shape)  # [B, T] [8, 19]
    print(batch["raw_texts"][:2]) # ['A bicycle figurine in which the front wheel is replaced with a clock', 'A black Honda motorcycle with a dark burgundy seat.']