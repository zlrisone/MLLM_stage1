import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from models.llm import QwenDecoder
from models.projector import LinearProjector

class MultimodalModel(nn.Module):
    """
    端到端多模态模型
    Vision Encoder + Projector + LLM Decoder
    """

    def __init__(self, config: dict):
        """
        初始化多模态模型

        Args:
            config: 完整配置字典
        """
        super().__init__()

        self.config = config

         # 构建各个组件
        self.vision_encoder = AutoModel.from_pretrained(config['model']['vision_encoder']['model_name'],trust_remote_code=True)
        self.llm_decoder = QwenDecoder(model_name=config['model']['llm']['model_name'])
        self.projector = LinearProjector(
            input_dim=config['model']['projector']['input_dim'],
            hidden_dim=config['model']['projector']['hidden_dim'],
            output_dim=config['model']['projector']['output_dim'],
            activation=config['model']['projector']['activation']
        )
        
        # 强制冻结
        for p in self.llm_decoder.parameters():
            p.requires_grad = False
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        
        # 模型参数统计
        self._log_model_info()
    def _log_model_info(self):
        """记录模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 50)
        print("Multimodal Model Summary")
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        ratio = 100 * trainable_params / total_params
        print(f"Trainable ratio: {ratio:.2f}%")
        print()

        # 各组件参数
        print("Component Parameters:")
        # print(f"  Vision Encoder: {self.vision_encoder.get_total_params():,} total, "
        #       f"{self.vision_encoder.get_trainable_params():,} trainable")
        print(f"  Projector: {self.projector.get_total_params():,} total, "
              f"{self.projector.get_trainable_params():,} trainable")
        print(f"  LLM Decoder: {self.llm_decoder.get_total_params():,} total, "
              f"{self.llm_decoder.get_trainable_params():,} trainable")
        print("=" * 50)
    def get_trainable_params(self) -> List[str]:
        """获取可训练参数名称列表"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        return trainable_params
    @staticmethod
    def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)

    @staticmethod
    def sigmoid_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
        """
        SigLIP pairwise sigmoid loss
        logits: (B, B)
        """
        bsz = logits.size(0)
        device = logits.device

        labels = torch.full((bsz, bsz), -1.0, device=device, dtype=logits.dtype)
        labels.fill_diagonal_(1.0)

        loss = F.softplus(-labels * logits).mean()
        return loss
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, H, W)
        return: (B, H)
        """
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values)   # (B, V, 768)

        projected = self.projector(vision_features)               # (B, V, H)
        image_embeds = projected.mean(dim=1)                     # (B, H)
        image_embeds = self.l2_normalize(image_embeds)
        return image_embeds
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, T)
        attention_mask: (B, T)
        return: (B, H)
        """
        with torch.no_grad():
            text_embeds = self.decoder.encode_text(input_ids, attention_mask)

        # (B, H)
        text_embeds = self.l2_normalize(text_embeds)
        return text_embeds
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        image_embeds = self.encode_image(pixel_values)           # (B, H)
        text_embeds = self.encode_text(input_ids, attention_mask) # (B, H)

        logits = torch.matmul(image_embeds, text_embeds.t())     # (B, B)
        # logits = self.logit_scale * logits + self.logit_bias

        loss = self.sigmoid_contrastive_loss(logits)

        return {
            "loss": loss,
            "logits": logits,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }

    def save_pretrained(self, save_path: str):
        """保存模型"""
        # 保存各个组件
        vision_path = f"{save_path}/vision_encoder"
        projector_path = f"{save_path}/projector"
        llm_path = f"{save_path}/llm_decoder"

        self.vision_encoder.vision_model.save_pretrained(vision_path)
        torch.save(self.projector.state_dict(), f"{projector_path}/pytorch_model.bin")

        if self.llm_decoder.use_lora:
            self.llm_decoder.model.save_pretrained(llm_path)
        else:
            self.llm_decoder.model.save_pretrained(llm_path)

        # 保存配置
        import json
        with open(f"{save_path}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

def create_multimodal_model(config: dict) -> MultimodalModel:
    """
    创建多模态模型

    Args:
        config: 配置字典

    Returns:
        多模态模型实例
    """
    return MultimodalModel(config)