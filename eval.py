import torch
import os
import random
import argparse
from dataclasses import dataclass, field
import transformers
from torch.utils.data import Dataset, DataLoader
import pdb
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file
from tqdm import tqdm
import random
import numpy as np
import torch
import cv2
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score

import torch.nn as nn

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
cache_dir = '/home/mila/s/soroush.omranpour/scratch/hf_cache'

class FakeClueDataset(Dataset):
    def __init__(self, data_dir, processor, train=False):
        super().__init__()
        self.data_dir = os.path.join(data_dir, 'train' if train else 'test')
        with open(os.path.join(data_dir, f'data_json/{"train" if train else "test"}.json') , 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx]['image'])
        label = 1 - self.data[idx]['label'] ## in the original dataset real=1 and fake=0
        
        image = Image.open(img_path)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        inputs = self.processor(image, return_tensors="pt")['pixel_values'][0]
        return inputs, label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained(
    "microsoft/swinv2-small-patch4-window16-256", cache_dir=cache_dir, use_fast=True
)


config = AutoConfig.from_pretrained("microsoft/swinv2-small-patch4-window16-256")
config.num_labels = 2
model = AutoModelForImageClassification.from_config(config)

model.load_state_dict(
    load_file("weights/model.safetensors"), 
    strict=False
)
model.cuda().eval()


dataset = FakeClueDataset(
    data_dir="/home/mila/s/soroush.omranpour/scratch/FakeClue", 
    processor=processor, 
    train=False
)
test_dataloader = DataLoader(
    dataset,
    batch_size=36,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)

from tqdm.auto import tqdm

gt = []
pred = []
with torch.inference_mode():
    for x, y in tqdm(test_dataloader):    
        pred += [model(x.to(device)).logits.argmax(-1).detach().cpu()]
        gt += [y]
    gt = torch.cat(gt)
    pred = torch.cat(pred)  


print(classification_report(gt, pred, digits=4))