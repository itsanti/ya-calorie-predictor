import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultimodalDataset(Dataset):
    def __init__(self, df, images_dir, missing_image_strategy="skip"):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.missing_image_strategy = missing_image_strategy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row.get("ingredients_text", "")
        mass = float(row["total_mass"])
        target = float(row["total_calories"])

        dish_id = row["dish_id"]
        img_path = os.path.join(self.images_dir, str(dish_id), "rgb.png")

        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except (FileNotFoundError, OSError) as e:
            if self.missing_image_strategy == "skip":
                return None
            elif self.missing_image_strategy == "zero":
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                raise ValueError(f"Unknown strategy: {self.missing_image_strategy}")

        return {
            "text": text,
            "image": image,
            "mass": mass,
            "target": target,
            "dish_id": dish_id
        }

def collate_fn(batch, tokenizer, image_transform):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.float)
    dish_ids = [item["dish_id"] for item in batch]

    masses = torch.tensor(
        [np.log1p(item["mass"]) for item in batch],
        dtype=torch.float
    ).unsqueeze(1)

    encoding = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )

    if image_transform:
        images = [image_transform(image=img)["image"] for img in images]

    images = torch.stack(images)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "image": images,
        "mass": masses,
        "target": targets,
        "dish_id": dish_ids
    }

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

def get_data(train_df, val_df, test_df, img_path, batch_size, tokenizer, num_workers=2):
    # Создаем лоадеры только если DF не None
    train_loader = None
    if train_df is not None:
        train_ds = MultimodalDataset(df=train_df, images_dir=img_path)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, tokenizer, train_transform)
        )

    val_loader = None
    if val_df is not None:
        val_ds = MultimodalDataset(df=val_df, images_dir=img_path)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, tokenizer, val_transform)
        )

    test_loader = None
    if test_df is not None:
        test_ds = MultimodalDataset(df=test_df, images_dir=img_path)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, tokenizer, val_transform)
        )

    return train_loader, val_loader, test_loader
