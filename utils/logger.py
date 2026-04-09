"""
日志记录模块
支持TensorBoard和WandB日志记录
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """统一日志记录器"""

    def __init__(self, log_dir: str, use_tensorboard: bool = True, use_wandb: bool = False,
                 wandb_project: str = "multimodal-llm", config: Optional[dict] = None):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录
            use_tensorboard: 是否使用TensorBoard
            use_wandb: 是否使用WandB
            wandb_project: WandB项目名称
            config: 配置字典
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # 初始化TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"TensorBoard logging enabled: {self.log_dir}")

        # 初始化WandB
        self.wandb_run = None
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project,
                config=config,
                dir=str(self.log_dir)
            )
            print(f"WandB logging enabled: {wandb_project}")

        # 保存配置
        if config is not None:
            config_path = self.log_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """
        记录指标

        Args:
            metrics: 指标字典
            step: 当前步骤
            prefix: 指标前缀
        """
        # 添加前缀
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics

        # TensorBoard记录
        if self.tb_writer is not None:
            for key, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # WandB记录
        if self.wandb_run is not None:
            self.wandb_run.log(prefixed_metrics, step=step)

    def log_text(self, text: str, step: int, key: str = "text"):
        """
        记录文本

        Args:
            text: 文本内容
            step: 当前步骤
            key: 文本键名
        """
        # TensorBoard记录
        if self.tb_writer is not None:
            self.tb_writer.add_text(key, text, step)

        # WandB记录
        if self.wandb_run is not None:
            self.wandb_run.log({key: text}, step=step)

    def log_image(self, image, step: int, key: str = "image"):
        """
        记录图像

        Args:
            image: 图像tensor或PIL图像
            step: 当前步骤
            key: 图像键名
        """
        # TensorBoard记录
        if self.tb_writer is not None:
            self.tb_writer.add_image(key, image, step)

        # WandB记录
        if self.wandb_run is not None:
            self.wandb_run.log({key: wandb.Image(image)}, step=step)

    def log_histogram(self, values, step: int, key: str = "histogram"):
        """
        记录直方图

        Args:
            values: 值tensor
            step: 当前步骤
            key: 直方图键名
        """
        # TensorBoard记录
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(key, values, step)

    def close(self):
        """关闭日志记录器"""
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TrainingLogger:
    """训练专用日志记录器"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.epoch_losses = []
        self.step_losses = []

    def log_train_step(self, loss: float, global_step: int, lr: Optional[float] = None, **kwargs) -> None:
        """
        记录单步训练信息

        Args:
            loss: 当前 loss
            global_step: 全局 step
            lr: 学习率
            **kwargs: 其他指标
        """
        metrics = {"loss": loss}
        if lr is not None:
            metrics["lr"] = lr
        metrics.update(kwargs)

        self.logger.log_metrics(metrics, global_step, prefix="train_step")
        self.step_losses.append(loss)

    def log_train_epoch(self, epoch: int, avg_loss: float, global_step: int, **kwargs) -> None:
        """
        记录 epoch 级训练信息

        Args:
            epoch: 当前 epoch
            avg_loss: 平均训练 loss
            global_step: 对应记录时的 global step
            **kwargs: 其他指标
        """
        metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
        }
        metrics.update(kwargs)

        self.logger.log_metrics(metrics, global_step, prefix="train_epoch")
        self.epoch_losses.append(avg_loss)

    def log_validation(self, epoch: int, metrics: Dict[str, float], global_step: int) -> None:
        """
        记录验证信息

        Args:
            epoch: 当前 epoch
            metrics: 验证指标
            global_step: 当前全局 step
        """
        payload = {"epoch": epoch}
        payload.update(metrics)
        self.logger.log_metrics(payload, global_step, prefix="val")

    def log_checkpoint(self, global_step: int, checkpoint_path: str, val_loss: Optional[float] = None) -> None:
        """
        记录检查点保存信息

        Args:
            global_step: 当前全局 step
            checkpoint_path: 检查点路径
            val_loss: 可选的验证损失
        """
        text = f"Checkpoint saved: {checkpoint_path}"
        if val_loss is not None:
            text += f" | val_loss={val_loss:.6f}"

        self.logger.log_text(text, global_step, key="checkpoint")
    def log_message(self, message: str, global_step: int, key: str = "message") -> None:
        """
        记录普通消息
        """
        self.logger.log_text(message, global_step, key=key)


def create_logger(log_dir: str, config: dict) -> Logger:
    """
    创建日志记录器

    Args:
        log_dir: 日志目录
        config: 配置字典

    Returns:
        日志记录器实例
    """
    use_tensorboard = config.get('logging', {}).get('tensorboard', True)
    use_wandb = config.get('logging', {}).get('wandb', False)
    wandb_project = config.get('logging', {}).get('wandb_project', 'multimodal-llm')

    return Logger(
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        config=config
    )


def create_training_logger(logger: Logger) -> TrainingLogger:
    """
    创建训练日志记录器

    Args:
        logger: 基础日志记录器

    Returns:
        训练日志记录器实例
    """
    return TrainingLogger(logger)