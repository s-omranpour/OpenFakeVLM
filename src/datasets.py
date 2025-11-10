import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class TestDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = []
        texts = []
        labels = []
        answers = []
        for sample in batch:
            images += [sample['image']]
            
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample['question']}]}]
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
            texts += [input_text]
            labels += [sample['label']]
            answers += [sample['answer']]
    
        inputs = self.tokenizer(
            images,
            texts,
            add_special_tokens = False,
            padding=True,
            return_tensors = "pt",
        )
        return inputs, labels, answers


class FakeClueSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', conversational=True):
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        json_path = os.path.join(data_dir, f'data_json/{split}.json')
        self.df = pd.read_json(json_path)
        self.df = self.df[(self.df.cate != 'doc') & (self.df.cate != 'satellite')]
        self.df.reset_index(inplace=True, drop=True)

        self.conversational = conversational

        self.user_key = 'human'
        self.assistant_key = 'gpt'


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        samp = self.df.loc[idx]
        label = 'fake' if samp['label'] == 0 else 'real'
        
        img_path = os.path.join(self.data_dir, samp['image'])
        image = Image.open(img_path)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        assert samp['conversations'][1]['from'] == 'gpt'
        question = 'Does the image look real/fake?'
        answer = samp['conversations'][1]['value'].strip()
        
        if self.conversational:
            messages = [
                {"role": "user", "content": [{"type": "image", "image" : image}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            return {'messages' : messages, 'images' : [image]}
        return {'image' : image, 'question' : question, 'answer' : answer, 'label' : label}