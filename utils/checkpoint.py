"""
检查点管理模块
负责模型保存、加载和恢复训练状态
"""

import os
import torch
import json
from typing import Dict, Any, Optional
from pathlib import Path


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器

        Args:
            output_dir: 检查点保存目录
            max_checkpoints: 最大保存检查点数量
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], epoch: int, step: int,
                       val_loss: float, config: dict, filename: str = None) -> str:
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            step: 当前step
            loss: 当前loss
            config: 配置字典
            filename: 自定义文件名

        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"checkpoint-epoch{epoch:03d}-step{step:06d}.pt"

        checkpoint_path = self.output_dir / filename

        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'config': config
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 保存检查点
        torch.save(checkpoint, checkpoint_path)

        # 保存配置文件
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint saved: {checkpoint_path}")

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)
    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
            model: 模型
            optimizer: 优化器 (可选)
            scheduler: 学习率调度器 (可选)

        Returns:
            检查点信息字典
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}, Loss: {checkpoint['val_loss']:.4f}")

        return checkpoint
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        查找最新的检查点

        Returns:
            最新检查点路径，如果不存在返回None
        """
        latest_path = self.output_dir / "checkpoint-latest.pt"
        if latest_path.exists():
            return str(latest_path)

        checkpoint_files = list(self.output_dir.glob("checkpoint-epoch*-step*.pt"))
        if not checkpoint_files:
            return None

        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoint_files[0])


    def find_best_checkpoint(self, metric: str = 'loss') -> Optional[str]:
        """
        查找最佳检查点

        Args:
            metric: 评估指标 ('loss', 'accuracy', etc.)

        Returns:
            最佳检查点路径
        """
        # 读取所有检查点的信息
        best_path = self.output_dir / "checkpoint-best.pt"
        if best_path.exists():
            return str(best_path)

        checkpoints_info = []

        for checkpoint_file in self.output_dir.glob("checkpoint-epoch*-step*.pt"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                val_loss = checkpoint.get("val_loss", checkpoint.get("loss", float("inf")))
                checkpoints_info.append(
                    {
                        "path": str(checkpoint_file),
                        "val_loss": val_loss,
                        "epoch": checkpoint.get("epoch", 0),
                        "step": checkpoint.get("step", 0),
                    }
                )
            except Exception as e:
                print(f"Failed to load checkpoint {checkpoint_file}: {e}")

        if not checkpoints_info:
            return None

        checkpoints_info.sort(key=lambda x: x["val_loss"])
        return checkpoints_info[0]["path"]

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，保持最大数量"""
        archive_files = list(self.output_dir.glob("checkpoint-epoch*-step*.pt"))

        if len(archive_files) <= self.max_checkpoints:
            return

        # 按修改时间排序，保留最新的
        # 删除旧的检查点
        archive_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for old_checkpoint in archive_files[self.max_checkpoints:]:
            try:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                print(f"Failed to remove checkpoint {old_checkpoint}: {e}")

    def save_final_model(self, model: torch.nn.Module, config: dict, filename: str = "final_model.pt"):
        """
        保存最终模型 (只保存模型权重，不包含优化器状态)

        Args:
            model: 模型
            config: 配置
            filename: 文件名
        """
        final_path = self.output_dir / filename

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, final_path)

        print(f"Final model saved: {final_path}")
        return str(final_path)


def create_checkpoint_manager(output_dir: str, max_checkpoints: int = 5) -> CheckpointManager:
    """
    创建检查点管理器

    Args:
        output_dir: 输出目录
        max_checkpoints: 最大检查点数量

    Returns:
        检查点管理器实例
    """
    return CheckpointManager(output_dir, max_checkpoints)