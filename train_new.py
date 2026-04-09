import argparse
import yaml
import os
import random
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.multimodal_model import create_multimodal_model
from data.caption_dataset import build_train_val_dataloaders
from utils.logger import create_logger, create_training_logger
from utils.checkpoint import create_checkpoint_manager

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """
    执行一次完整验证

    Args:
        model: 模型
        dataloader: 验证集 dataloader
        device: 设备
        epoch: 当前 epoch

    Returns:
        平均验证 loss
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Val   Epoch {epoch + 1:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

    return total_loss / max(num_batches, 1)

def train(model: nn.Module, train_loader : DataLoader, val_loader : DataLoader, 
    optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
    training_logger,start_epoch:int, num_epochs: int, config: dict,best_val_loss:float,checkpoint_manager):
    
    for epoch in range(start_epoch, num_epochs):

        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1:02d}",
            leave=True,
            dynamic_ncols=True,
        )
        
        for step, batch in enumerate(pbar):
            global_step = epoch * len(train_loader) + step

            # 移动数据到设备
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = outputs["loss"]

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            num_batches += 1
            avg_loss = total_loss / num_batches

            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

            # 日志记录
            if global_step % config['logging_steps'] == 0:
                training_logger.log_step(loss.item(), global_step, current_lr)
            
            # 验证和保存检查点
            if global_step % config['eval_steps'] == 0 and global_step > 0:
                val_loss = validate(model, val_loader, device, epoch)
                print(f"[Eval @ step {global_step}] val_loss={val_loss:.4f}")
                training_logger.log_validation(
                    epoch=epoch,
                    metrics={"loss": val_loss},
                    global_step=global_step,
                )
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step, val_loss, config,
                        filename="checkpoint-best.pt"
                    )
                    training_logger.log_checkpoint(
                        global_step=global_step,
                        checkpoint_path="checkpoint-best.pt",
                        val_loss=val_loss,
                    )
                
                model.train()
        epoch_avg_loss = total_loss / max(num_batches, 1)
        end_of_epoch_step = (epoch + 1) * len(train_loader) - 1

        training_logger.log_train_epoch(
            epoch=epoch,
            avg_loss=epoch_avg_loss,
            global_step=end_of_epoch_step,
        )
        # 每个 epoch 结束后做一次验证
        val_loss = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_avg_loss={epoch_avg_loss:.4f} | val_loss={val_loss:.4f}"
        )

        training_logger.log_validation(
            epoch=epoch,
            metrics={"loss": val_loss},
            global_step=end_of_epoch_step,
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=end_of_epoch_step,
                val_loss=val_loss,
                config=config,
                filename="checkpoint-best.pt",
            )
            training_logger.log_checkpoint(
                global_step=end_of_epoch_step,
                checkpoint_path="checkpoint-best.pt",
                val_loss=val_loss,
            )
        # 保存检查点
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=end_of_epoch_step,
            val_loss=val_loss,
            config=config,
        )
        training_logger.log_checkpoint(
            global_step=end_of_epoch_step,
            checkpoint_path=checkpoint_path,
            val_loss=val_loss,
        )
    # 保存最终模型
    checkpoint_manager.save_final_model(model, config)
    return best_val_loss
def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training: Alignment")
    parser.add_argument("--config", type=str, default="./config/training_stage1.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(config['seed'])
    
    # 创建日志记录器
    logger = create_logger(config['logging_dir'], config)
    training_logger = create_training_logger(logger)
    
    try:
        # 创建模型
        model = create_multimodal_model(config)
        model.to(device)
        # 创建数据
        train_dataset, val_dataset, train_loader, val_loader = build_train_val_dataloaders(
            vision_model_name=config['model']['vision_encoder']['model_name'],
            qwen_model_name=config['model']['llm']['model_name'],
            split="val",
            train_ratio=config['dataset']['train_ratio'],
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            max_length=config['dataset']['max_length'],
            sample_one_caption=True,
            add_prompt=False,
            seed=config['seed'],
        )
        # 创建优化器
        optimizer_config = config['optimizer']
        if optimizer_config['name'] == 'adamw':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay'],
                betas=tuple(optimizer_config['betas'])
            )

        num_epochs = config['epochs']
        # 创建学习率调度器
        scheduler_config = config['scheduler']
        if scheduler_config['name'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs * len(train_loader),
                eta_min=0
            )
        else:
            scheduler = None
        
        # 创建检查点管理器
        checkpoint_manager = create_checkpoint_manager(config['output_dir'])

        # 恢复检查点 (如果指定)
        start_epoch = 0
        best_val_loss = float('inf')
       
        if args.resume:
            checkpoint = checkpoint_manager.load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(
                f"Resumed from checkpoint: {args.resume} | "
                f"start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}"
            )
        
        

        best_val_loss = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            training_logger=training_logger,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            config=config,
            best_val_loss=best_val_loss,
            checkpoint_manager=checkpoint_manager,
        )

        print(f"Training finished. Best val loss: {best_val_loss:.6f}")
    finally:
        logger.close()
if __name__ == "__main__":
    main()