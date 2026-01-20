#!/usr/bin/env python3
"""Generate segmentation masks from COCO annotations with correct rendering order."""
import argparse
import json
import os
import numpy as np
import cv2
from pycocotools import mask as coco_mask


def polygon_to_mask(seg, h, w):
    """Convert polygon segmentation to binary mask."""
    if isinstance(seg, dict):
        # RLE format
        rle = seg
        if isinstance(rle.get("counts"), str):
            rle = {"counts": rle["counts"].encode("utf-8"), "size": [h, w]}
        return coco_mask.decode(rle)
    else:
        # Polygon format
        rles = coco_mask.frPyObjects(seg, h, w)
        rle = coco_mask.merge(rles)
        return coco_mask.decode(rle)


def generate_masks(dataset_path, output_dir, selections_path=None, viz_dir=None, excluded_path=None):
    """Generate segmentation masks with correct rendering order."""
    ann_path = os.path.join(dataset_path, "instances_default.json")
    with open(ann_path) as f:
        data = json.load(f)

    # Load selections (grasping frames)
    grasp_frames = set()
    if selections_path and os.path.exists(selections_path):
        with open(selections_path) as f:
            selections = json.load(f)
        grasp_frames = set(selections.get("marks", {}).keys())
        print(f"Loaded {len(grasp_frames)} grasping frames from selections.json")

    # Load excluded frames
    excluded_frames = set()
    if excluded_path and os.path.exists(excluded_path):
        with open(excluded_path) as f:
            excluded = json.load(f)
        for fname, status in excluded.get("marks", {}).items():
            if status == "Excluded":
                # Convert .png to .jpg if needed
                base = os.path.splitext(fname)[0]
                excluded_frames.add(base + ".jpg")
                excluded_frames.add(base + ".png")
        print(f"Loaded {len(excluded_frames)//2} excluded frames from excluded.json")

    # Build category mapping
    cats = {c["id"]: c["name"] for c in data["categories"]}
    cat_to_id = {v: k for k, v in cats.items()}
    print(f"Categories: {cats}")

    # Find static_finger category
    static_finger_id = cat_to_id.get("static_finger")
    if static_finger_id is None:
        raise ValueError("No static_finger category found")

    # Group annotations by image
    anns_by_img = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    # Build image id to info mapping
    id_to_img = {img["id"]: img for img in data["images"]}

    # Filter to images with static_finger
    valid_imgs = []
    for img_id, anns in anns_by_img.items():
        has_finger = any(a["category_id"] == static_finger_id for a in anns)
        if has_finger:
            valid_imgs.append(img_id)

    print(f"Images with static_finger: {len(valid_imgs)} / {len(data['images'])}")

    os.makedirs(output_dir, exist_ok=True)
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    # Colors for visualization (BGR)
    colors = {
        1: (0, 255, 0),      # background - green
        2: (255, 0, 255),    # table - magenta
        3: (0, 127, 255),    # cube - orange
        4: (255, 127, 0),    # static_finger - cyan
        5: (127, 255, 127),  # moving_finger - light green
    }

    # Define rendering orders
    # Default: bg(1) -> table(2) -> cube(3) -> static_finger(4) -> moving_finger(5)
    # Grasping: bg(1) -> table(2) -> static_finger(4) -> moving_finger(5) -> cube(3)
    default_order = [
        cat_to_id.get("background"),
        cat_to_id.get("table"),
        cat_to_id.get("cube"),
        cat_to_id.get("static_finger"),
        cat_to_id.get("moving_finger"),
    ]
    grasp_order = [
        cat_to_id.get("background"),
        cat_to_id.get("table"),
        cat_to_id.get("static_finger"),
        cat_to_id.get("moving_finger"),
        cat_to_id.get("cube"),
    ]
    # Filter None values
    default_order = [x for x in default_order if x is not None]
    grasp_order = [x for x in grasp_order if x is not None]

    processed = 0
    for img_id in valid_imgs:
        img_info = id_to_img[img_id]
        filename = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        # Skip excluded frames
        if filename in excluded_frames:
            continue

        # Determine rendering order
        is_grasp = filename in grasp_frames
        render_order = grasp_order if is_grasp else default_order

        # Create segmentation mask (start with zeros = background label 0)
        seg_mask = np.zeros((h, w), dtype=np.uint8)

        # Get annotations for this image grouped by category
        anns = anns_by_img[img_id]
        anns_by_cat = {}
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in anns_by_cat:
                anns_by_cat[cat_id] = []
            anns_by_cat[cat_id].append(ann)

        # Render in order (later overwrites earlier)
        for cat_id in render_order:
            if cat_id not in anns_by_cat:
                continue
            for ann in anns_by_cat[cat_id]:
                mask = polygon_to_mask(ann["segmentation"], h, w)
                seg_mask[mask > 0] = cat_id

        # Save mask
        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, seg_mask)

        # Save visualization
        if viz_dir:
            img_path = os.path.join(dataset_path, "images", "default", filename)
            if not os.path.exists(img_path):
                img_path = os.path.join(dataset_path, "images", filename)
            img = cv2.imread(img_path)
            if img is not None:
                viz = img.copy()
                for cat_id, color in colors.items():
                    mask = (seg_mask == cat_id)
                    viz[mask] = (0.5 * viz[mask] + 0.5 * np.array(color)).astype(np.uint8)
                viz_path = os.path.join(viz_dir, out_name)
                cv2.imwrite(viz_path, viz)

        processed += 1

        if processed % 20 == 0:
            print(f"Processed {processed}/{len(valid_imgs)}")

    # Count processed frames by type
    processed_imgs = [img_id for img_id in valid_imgs if id_to_img[img_id]["file_name"] not in excluded_frames]
    grasp_count = len([f for f in processed_imgs if id_to_img[f]['file_name'] in grasp_frames])
    default_count = len(processed_imgs) - grasp_count

    print(f"\nSaved {processed} masks to {output_dir}/")
    print(f"  - Default order (cube behind fingers): {default_count}")
    print(f"  - Grasp order (cube in front): {grasp_count}")
    if excluded_frames:
        print(f"  - Excluded: {len(valid_imgs) - len(processed_imgs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--selections", type=str, default=None,
                        help="Path to selections.json for grasping frames")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Output directory for visualization images")
    parser.add_argument("--excluded", type=str, default=None,
                        help="Path to excluded.json for frames to skip")
    args = parser.parse_args()

    generate_masks(args.dataset_path, args.output_dir, args.selections, args.viz_dir, args.excluded)
