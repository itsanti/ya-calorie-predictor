import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import pandas as pd
import numpy as np
import os
from src.dataset import get_data
from src.model import MultimodalModel
from sklearn.metrics import r2_score

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in dataloader:
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        masses = batch["mass"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        batch_size = targets.size(0)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images, mass=masses)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images, mass=masses)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_mae  += F.l1_loss(outputs, targets, reduction="sum").item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_mae  = total_mae / total_samples if total_samples > 0 else 0

    return avg_loss, avg_mae

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    all_targets, all_preds = [], []

    for batch in dataloader:
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        masses = batch["mass"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        batch_size = targets.size(0)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images, mass=masses)
                loss = criterion(outputs, targets)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images, mass=masses)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size
        total_mae  += F.l1_loss(outputs, targets, reduction="sum").item()
        total_samples += batch_size

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(outputs.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_mae  = total_mae / total_samples if total_samples > 0 else 0

    return avg_loss, avg_mae, all_targets, all_preds


def train(config, train_df, val_df, tokenizer):
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_data(
        train_df, val_df, None,
        img_path=config.IMAGES_PATH,
        batch_size=config.BATCH_SIZE,
        tokenizer=tokenizer
    )

    model = MultimodalModel(
        text_unfreeze=config.TEXT_MODEL_UNFREEZE,
        image_unfreeze=config.IMAGE_MODEL_UNFREEZE,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.text_proj.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_proj.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.cross_attention.parameters(), 'lr': config.TEXT_LR},
        {'params': model.text_attention.parameters(), 'lr': config.TEXT_LR},
        {'params': model.film.parameters(), 'lr': config.TEXT_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR}
    ], weight_decay=0.01)

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=2)
    scaler = torch.amp.GradScaler(
        'cuda') if config.USE_AMP and device.type == 'cuda' else None

    best_val_mae = float('inf')
    patience_counter = 0

    # ОБНОВЛЕННАЯ ИСТОРИЯ
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': []  # Добавили поле для R^2
    }

    for epoch in range(config.EPOCHS):
        print(f"\n{'=' * 60}")
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Получаем предсказания и таргеты для расчета R2
        val_loss, val_mae, val_targets, val_preds = validate_one_epoch(
            model, val_loader, criterion, device, scaler
        )

        # Расчет R2
        current_r2 = r2_score(val_targets, val_preds)
        scheduler.step(val_loss)

        # Сохранение в историю
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(current_r2)

        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}")
        print(
            f"Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.2f} | Val R2: {current_r2:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'history': history  # Полезно сохранить и историю тоже
            }, config.SAVE_PATH)
            print(f"✓ New best model saved (MAE: {val_mae:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered")
                break

    return model, history
