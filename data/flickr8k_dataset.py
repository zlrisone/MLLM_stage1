import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, AutoTokenizer

import random
from typing import Dict, List, Optional,Any

class flickrCollator:
    """
    将 flickr8k 的样本整理成:
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
        answers = example["caption_1"]

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
def build_flickr_dataloader(
    vision_model_name: str,
    qwen_model_name: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 64,
    shuffle: bool = True,
    sample_one_caption: bool = True,
    add_prompt: bool = False,
    streaming: bool = False,
):
    """
    构建 flickr 的 DataLoader

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
        "jxie/flickr8k",
        split = split
    )
    collator = flickrCollator(
        vision_model_name=vision_model_name,
        qwen_model_name=qwen_model_name,
        max_length=max_length,
        sample_one_caption=sample_one_caption,
        add_prompt=add_prompt,
    )
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