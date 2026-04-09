import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, Any, List
import torch.nn.functional as F

class QwenDecoder(nn.Module):
    """
    Qwen2.5语言模型解码器包装器
    支持Lora微调和全量微调
    """
    def __init__(self, model_name:str="Qwen/Qwen2.5-3B", freeze:bool=True, use_lora:bool=False, lora_config:Optional[dict]=None):
        """
        初始化Qwen解码器
        Args:
            model_name: Qwen模型名称
            freeze: 是否冻结权重 (全量微调时为False)
            use_lora: 是否使用LoRA
            lora_config: LoRA配置字典
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.use_lora = use_lora

        # 加载预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map = None,
            trust_remote_code=True,
        )

        # 获取模型配置
        self.config = self.model.config
        self.vocab_size = self.config.vocab_size # 151936
        self.hidden_size = self.config.hidden_size # 2048

        # 应用LoRA
        if use_lora:
            self._apply_lora(lora_config)
        
        # 冻结参数
        if freeze and not use_lora:
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"Qwen Decoder loaded: {model_name}")
        print(f"Hidden size: {self.hidden_size}, Vocab size: {self.vocab_size}")
        print(f"Freeze: {freeze}, LoRA: {use_lora}")
        print(f"Trainable params: {self.get_trainable_params():,}")
    
    def _apply_lora(self, lora_config:dict):
        """应用LoRA配置"""
        # 默认LoRA配置
        default_config = {
            'r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'lora_layers': None  # None表示所有层
        }

        # 合并配置
        if lora_config:
            default_config.update(lora_config)

        # 构建LoRA配置
        lora_config = LoraConfig(
            r=default_config['r'],
            lora_alpha=default_config['lora_alpha'],
            lora_dropout=default_config['lora_dropout'],
            target_modules=default_config['target_modules'],
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)

        # 如果指定了层数，只在最后N层应用LoRA
        if default_config['lora_layers'] is not None:
            self._apply_lora_to_layers(default_config['lora_layers'])

    def _apply_lora_to_layers(self, num_layers:int):
        """只在最后N层应用LoRA"""
        # 获取模型的所有层
        layers = self.model.base_model.model.layers

        # 计算要应用LoRA的层索引
        total_layers = len(layers)
        start_layer = max(0, total_layers - num_layers)

        print(f"Applying LoRA to layers {start_layer} to {total_layers-1}")

        # 冻结前面层的LoRA参数
        for i in range(start_layer):
            for name, param in layers[i].named_parameters():
                if 'lora_' in name:
                    param.requires_grad = False
    @staticmethod
    def masked_mean_pooling(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对 hidden states 做 masked mean pooling

        Args:
            hidden_states: (B, seq_len, hidden_size)
            attention_mask: (B, seq_len), 1 表示有效 token, 0 表示 padding

        Returns:
            pooled: (B, hidden_size)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)   # (B, seq_len, 1)
        masked_hidden = hidden_states * mask                          # (B, seq_len, hidden_size)
        token_count = mask.sum(dim=1).clamp(min=1e-6)                # (B, 1)
        pooled = masked_hidden.sum(dim=1) / token_count              # (B, hidden_size)
        return pooled

    def forward(self, input_ids:torch.Tensor, attention_mask: Optional[torch.tensor] = None, labels : Optional[torch.tensor]=None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ids (B, seq_len)
            attention_mask: 注意力掩码 (B, seq_len)
            labels: 标签 (用于计算loss) (B, seq_len)
            **kwargs: 其他参数

        Returns:
            模型输出字典
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,   # 必须打开
            return_dict=True,
            **kwargs
        )
        # hidden_states:
        #   hidden_states[0]   -> embedding output
        #   hidden_states[-1]  -> 最后一层
        #   hidden_states[-2]  -> 倒数第二层
        penultimate_hidden_states = outputs.hidden_states[-2]  # (B, seq_len, hidden_size)

        penultimate_mean_pooled = self.masked_mean_pooling(
            penultimate_hidden_states,
            attention_mask
        )  # (B, hidden_size)
        return {
            'logits': outputs.logits,  # (B, seq_len, vocab_size)
            'loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'penultimate_mean_pooled': penultimate_mean_pooled         # (B, hidden_size)
        }
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                max_new_tokens: int = 100, **kwargs) -> torch.Tensor:
        """
        生成文本

        Args:
            input_ids: 输入token ids (B, seq_len)
            attention_mask: 注意力掩码 (B, seq_len)
            max_new_tokens: 最大生成token数
            **kwargs: 生成参数

        Returns:
            生成的token ids (B, seq_len + generated)
        """
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                **kwargs
            )

        return generated_ids
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        penultimate_hidden_states = outputs.hidden_states[-2]
        penultimate_mean_pooled = self.masked_mean_pooling(
            penultimate_hidden_states,
            attention_mask
        )
        return penultimate_mean_pooled
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_path: str):
        """保存模型"""
        if self.use_lora:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)

    def load_adapter(self, adapter_path: str):
        """加载LoRA适配器"""
        if self.use_lora:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def merge_and_unload(self):
        """合并LoRA权重并卸载LoRA模块"""
        if self.use_lora:
            self.model = self.model.merge_and_unload()
            self.use_lora = False
