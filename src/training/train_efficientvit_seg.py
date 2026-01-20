#!/usr/bin/env python3
"""Fine-tune EfficientViT-B0 for semantic segmentation using PyTorch Lightning."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics import JaccardIndex
from torchvision import transforms


class SoftIoULoss(nn.Module):
    """Soft IoU loss for semantic segmentation."""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with image/mask pairs."""

    def __init__(self, image_paths, mask_paths, img_height=480, img_width=640, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment

        # Color augmentations (only applied to image, not mask)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = Image.fromarray(mask)

        if self.augment:
            # Random resized crop (scale 0.7-1.0, ratio 0.9-1.1)
            if np.random.rand() > 0.3:
                # Get random crop parameters
                scale = np.random.uniform(0.7, 1.0)
                ratio = np.random.uniform(0.9, 1.1)
                orig_w, orig_h = img.size

                # Calculate crop size
                area = orig_w * orig_h * scale
                new_w = int(np.sqrt(area * ratio))
                new_h = int(np.sqrt(area / ratio))
                new_w = min(new_w, orig_w)
                new_h = min(new_h, orig_h)

                # Random crop position
                left = np.random.randint(0, orig_w - new_w + 1)
                top = np.random.randint(0, orig_h - new_h + 1)

                # Apply same crop to both
                img = img.crop((left, top, left + new_w, top + new_h))
                mask = mask.crop((left, top, left + new_w, top + new_h))

            # Horizontal flip
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Vertical flip
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # Random rotation (-15 to +15 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                img = img.rotate(angle, Image.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, Image.NEAREST, fillcolor=0)

            # Color jitter (image only)
            img = self.color_jitter(img)

            # Random Gaussian blur
            if np.random.rand() > 0.7:
                img = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(img)

        # Resize to target size
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        mask = mask.resize((self.img_width, self.img_height), Image.NEAREST)

        img_tensor = transforms.ToTensor()(img)
        img_tensor = self.normalize(img_tensor)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return img_tensor, mask_tensor


class EfficientViTSegModule(pl.LightningModule):
    """PyTorch Lightning module for EfficientViT segmentation."""

    def __init__(
        self,
        num_classes: int = 6,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_height: int = 480,
        img_width: int = 640,
        loss_type: str = "iou",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Use timm EfficientViT backbone (more flexible for small inputs)
        self.backbone = timm.create_model(
            "efficientvit_b0.r224_in1k",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Multi-scale features
        )

        # Get feature channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_height, img_width)
            feats = self.backbone(dummy)
            self.feat_channels = [f.shape[1] for f in feats]

        # Simple FPN-like decoder
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, 64, 1) for ch in self.feat_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ) for _ in self.feat_channels
        ])
        self.seg_head = nn.Conv2d(64, num_classes, 1)

        if loss_type == "iou":
            self.criterion = SoftIoULoss(num_classes)
        elif loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)

        # FPN-style aggregation (top-down)
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )

        # Output convs
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Use finest resolution feature
        out = outs[0]
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        out = self.seg_head(out)
        return out

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        self.train_iou(preds, masks)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/iou", self.train_iou, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        self.val_iou(preds, masks)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/iou", self.val_iou, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def create_stratified_split(dataset_path, selections_path, excluded_path, val_ratio=0.15, seed=42):
    """Create stratified train/val split based on grasp vs default frames."""
    masks_dir = Path(dataset_path) / "masks"
    images_dir = Path(dataset_path) / "images" / "default"
    if not images_dir.exists():
        images_dir = Path(dataset_path) / "images"

    grasp_frames = set()
    if selections_path and os.path.exists(selections_path):
        with open(selections_path) as f:
            grasp_frames = set(json.load(f).get("marks", {}).keys())

    excluded_frames = set()
    if excluded_path and os.path.exists(excluded_path):
        with open(excluded_path) as f:
            for fname, status in json.load(f).get("marks", {}).items():
                if status == "Excluded":
                    excluded_frames.add(os.path.splitext(fname)[0])

    image_paths, mask_paths, labels = [], [], []
    for mask_path in sorted(masks_dir.glob("*.png")):
        base_name = mask_path.stem
        if base_name in excluded_frames:
            continue
        img_path = images_dir / f"{base_name}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{base_name}.png"
        if not img_path.exists():
            continue
        image_paths.append(img_path)
        mask_paths.append(mask_path)
        labels.append(1 if any(base_name in gf for gf in grasp_frames) else 0)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_ratio, stratify=labels, random_state=seed
    )
    print(f"Total: {len(image_paths)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")
    print(f"Grasp: {sum(labels)} | Default: {len(labels) - sum(labels)}")
    return train_imgs, val_imgs, train_masks, val_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="data/dataset_v0")
    parser.add_argument("--selections", type=str, default="data/dataset_v0/selections.json")
    parser.add_argument("--excluded", type=str, default="data/dataset_v0/excluded.json")
    parser.add_argument("--output-dir", type=str, default="outputs/efficientvit_seg")
    parser.add_argument("--img-height", type=int, default=480)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)  # Reduced for 640x480
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--loss", type=str, default="iou", choices=["iou", "ce"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    train_imgs, val_imgs, train_masks, val_masks = create_stratified_split(
        args.dataset_path, args.selections, args.excluded, val_ratio=0.15, seed=args.seed
    )

    train_dataset = SegmentationDataset(
        train_imgs, train_masks, img_height=args.img_height, img_width=args.img_width, augment=True
    )
    val_dataset = SegmentationDataset(
        val_imgs, val_masks, img_height=args.img_height, img_width=args.img_width, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = EfficientViTSegModule(
        num_classes=args.num_classes, lr=args.lr,
        img_height=args.img_height, img_width=args.img_width, loss_type=args.loss
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best",
        monitor="val/iou",
        mode="max",
        save_top_k=1,
    )
    callbacks = [
        checkpoint_callback,
        pl.callbacks.EarlyStopping(monitor="val/iou", patience=20, mode="max"),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1,
        callbacks=callbacks, default_root_dir=args.output_dir, log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest model: {checkpoint_callback.best_model_path}")
    print(f"Best val/iou: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
