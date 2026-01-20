#!/usr/bin/env python3
"""Inference script for EfficientViT segmentation model."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from src.training.train_efficientvit_seg import EfficientViTSegModule
except ModuleNotFoundError:
    from train_efficientvit_seg import EfficientViTSegModule


CLASS_COLORS = {
    0: (0, 0, 0),        # unlabeled - black
    1: (0, 255, 0),      # background - green
    2: (255, 0, 255),    # table - magenta
    3: (0, 127, 255),    # cube - orange
    4: (255, 127, 0),    # static_finger - cyan
    5: (127, 255, 127),  # moving_finger - light green
}

CLASS_NAMES = {
    0: "unlabeled",
    1: "background",
    2: "table",
    3: "cube",
    4: "static_finger",
    5: "moving_finger",
}


class SegmentationInference:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = EfficientViTSegModule.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Support both old (img_size) and new (img_height/img_width) checkpoint formats
        if hasattr(self.model.hparams, 'img_height'):
            self.img_height = self.model.hparams.img_height
            self.img_width = self.model.hparams.img_width
        else:
            self.img_height = self.model.hparams.img_size
            self.img_width = self.model.hparams.img_size
        self.num_classes = self.model.hparams.num_classes

        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict segmentation mask from BGR image.

        Args:
            image: BGR image (H, W, 3) numpy array

        Returns:
            Segmentation mask (H, W) with class IDs
        """
        orig_h, orig_w = image.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize back to original size
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay segmentation mask on image.

        Args:
            image: BGR image (H, W, 3)
            mask: Segmentation mask (H, W)
            alpha: Blend factor

        Returns:
            Visualization image (H, W, 3)
        """
        viz = image.copy()
        for class_id, color in CLASS_COLORS.items():
            class_mask = (mask == class_id)
            viz[class_mask] = (
                (1 - alpha) * viz[class_mask] + alpha * np.array(color)
            ).astype(np.uint8)
        return viz


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def process_video(inferencer, input_path: Path, output_path: str, alpha: float):
    """Process video file and save with segmentation overlay."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = inferencer.predict(frame)
        viz = inferencer.visualize(frame, mask, alpha)
        out.write(viz)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"\nSaved video: {output_path} ({frame_idx} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/efficientvit_seg/best.ckpt")
    parser.add_argument("--input", type=str, required=True, help="Input image, video, or directory")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=0.5, help="Visualization blend factor")
    parser.add_argument("--save-mask", action="store_true", help="Save raw mask in addition to visualization")
    args = parser.parse_args()

    inferencer = SegmentationInference(args.checkpoint, args.device)
    input_path = Path(args.input)

    # Check if input is video
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        output_path = args.output if args.output else f"{input_path.stem}_seg.mp4"
        process_video(inferencer, input_path, output_path, args.alpha)

    elif input_path.is_dir():
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        output_dir = Path(args.output) if args.output else input_path / "seg_output"
        output_dir.mkdir(exist_ok=True)

        for img_path in images:
            image = cv2.imread(str(img_path))
            mask = inferencer.predict(image)
            viz = inferencer.visualize(image, mask, args.alpha)

            out_path = output_dir / f"{img_path.stem}_seg.png"
            cv2.imwrite(str(out_path), viz)

            if args.save_mask:
                mask_path = output_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask)

            print(f"Processed: {img_path.name}")

        print(f"\nSaved {len(images)} results to {output_dir}/")
    else:
        image = cv2.imread(str(input_path))
        mask = inferencer.predict(image)
        viz = inferencer.visualize(image, mask, args.alpha)

        output_path = args.output if args.output else f"{input_path.stem}_seg.png"
        cv2.imwrite(output_path, viz)
        print(f"Saved: {output_path}")

        if args.save_mask:
            mask_path = f"{input_path.stem}_mask.png"
            cv2.imwrite(mask_path, mask)
            print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
    main()
