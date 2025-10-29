#!/usr/bin/env python
import argparse
import os
import torch

from transformers import AutoModel


def extract(model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
    )

    sd = model.state_dict()

    sam = {}
    clip = {}
    projector = {}

    for k, v in sd.items():
        if 'sam_model' in k:
            sam[k.replace('model.', '', 1)] = v
        elif 'vision_model' in k:
            clip[k.replace('model.', '', 1)] = v
        elif 'projector' in k:
            projector[k.replace('model.', '', 1)] = v

    torch.save(sam, os.path.join(output_dir, 'sam_encoder.pth'))
    torch.save(clip, os.path.join(output_dir, 'clip_encoder.pth'))
    torch.save(projector, os.path.join(output_dir, 'projector.pth'))

    print(f"Saved: {len(sam)} sam, {len(clip)} clip, {len(projector)} projector params -> {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-OCR')
    parser.add_argument('--out', default='encoder_weights')
    args = parser.parse_args()
    extract(args.model, args.out)


if __name__ == '__main__':
    main()


