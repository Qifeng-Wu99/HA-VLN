# from groundingdino.util.inference import load_model, load_image, predict, annotate, preprocess_caption
import cv2
import torch
from torch import nn
from groundingdino.util.utils import get_phrases_from_posmap
import bisect


import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_pil):
    # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image
    image_pil = Image.fromarray(np.array(image_pil))
    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = load_model("path/to/cofig", "path/to/checkpoint")
        import os
        current_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_path)
        self.model = load_model(os.path.join(current_dir, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                                 os.path.join(current_dir,"GroundingDINO/weights/groundingdino_swint_ogc.pth"))
        self.box_threshold = 0.35
        self.text_threshold = 0.25
    
    def forward(self,
        observations,
        caption: str,
        current_episodes,
        stats_info,
        device: str = "cuda",
        remove_combined: bool = False
        ):
        # caption = preprocess_caption(caption=caption)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        model = self.model.to(device)
        image = [ob['rgb'] for ob in observations]
        images_pil, images = [], []
        for im in image:
            image_pil, image = load_image(im)
            images_pil.append(image_pil)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=[caption]*len(observations))
        
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (nq, 4)

        count = []
        images_with_box = []
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        for k in range(len(observations)):
            prediction_logits_single = prediction_logits[k]
            prediction_boxes_single = prediction_boxes[k]

            mask = prediction_logits_single.max(dim=1)[0] > self.box_threshold
            logits = prediction_logits_single[mask]  # logits.shape = (n, 256)
            boxes = prediction_boxes_single[mask]  # boxes.shape = (n, 4)
            
            if remove_combined:
                sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
                
                phrases = []
                for logit in logits:
                    max_idx = logit.argmax()
                    insert_idx = bisect.bisect_left(sep_idx, max_idx)
                    right_idx = sep_idx[insert_idx]
                    left_idx = sep_idx[insert_idx - 1]
                    phrases.append(get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
            else:
                phrases = [
                    get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer).replace('.', '')
                    for logit
                    in logits
                ]
            
            size = images_pil[k].size
            pred_dict = {
                "boxes": boxes,
                "size": [size[1], size[0]],  # H,W
                "labels": phrases,
            }
            count.append(len(phrases))
            if current_episodes[k].episode_id not in stats_info.keys():
                stats_info[str(current_episodes[k].episode_id)] = {}
                stats_info[str(current_episodes[k].episode_id)]['human_counting'] = []
            stats_info[str(current_episodes[k].episode_id)]['human_counting'].append(len(phrases))
            # import ipdb; ipdb.set_trace()
            image_with_box = plot_boxes_to_image(images_pil[k], pred_dict)[0]
            image_with_box  = np.asarray(image_with_box)
            images_with_box.append(image_with_box)

        return images_with_box