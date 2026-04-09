import torch
import torch.nn as nn
class LinearProjector(nn.Module):
    """
    线性投影器: 视觉特征 -> 语言模型嵌入空间
    架构: Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, output_dim)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1536,
                 output_dim: int = 2048, activation: str = "gelu"):
        """
        初始化投影器

        Args:
            input_dim: 输入维度 (视觉特征维度)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (LLM嵌入维度)
            activation: 激活函数类型
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 第一层线性变换
        self.linear1 = nn.Linear(input_dim, hidden_dim)

        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 第二层线性变换
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # 初始化权重
        self._init_weights()

        print(f"Linear Projector: {input_dim} -> {hidden_dim} -> {output_dim}")
        print(f"Trainable params: {self.get_trainable_params():,}")

    def _init_weights(self):
        """初始化权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

        # 偏置初始化为0
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            vision_features: 视觉特征 (B, seq_len, input_dim)
                             通常seq_len=197 (196 patches + 1 cls)

        Returns:
            投影后的特征 (B, seq_len, output_dim)
        """
        # 第一层: 线性变换 + 激活
        hidden = self.activation(self.linear1(vision_features))

        # 第二层: 线性变换
        output = self.linear2(hidden)

        return output

    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())