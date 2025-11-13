import pandas as pd
import numpy as np
import os
from PIL import Image
import torch


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


class FakeCluePromptDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, split='train', max_image_size=512, max_num_samples=None):
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        json_path = os.path.join(data_dir, f'data_json/{split}.json')
        self.df = pd.read_json(json_path)
        self.df = self.df[(self.df.cate != 'doc') & (self.df.cate != 'satellite')]
        self.df.reset_index(inplace=True, drop=True)
        if max_num_samples is not None:
            self.df = self.df[:max_num_samples]
            
        self.df['label'] = self.df['label'].apply(lambda x: 'fake' if x == 0 else 'real')
        self.df['answer'] = self.df['conversations'].apply(lambda x: x[1]['value'].strip())
        self.df['answer'] = self.df['answer'].apply(lambda x: '. '.join(x.split('. ')[1:]))
        
        self.tokenizer = tokenizer
        self.max_image_size = max_image_size


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        samp = self.df.loc[idx]
        label = samp['label']
        question = 'Does the image look real/fake?'
        
        img_path = os.path.join(self.data_dir, samp['image'])
        image = Image.open(img_path)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        h, w = image.size
        m = max(h, w)
        if m > self.max_image_size:
            h = int(h/m * self.max_image_size)
            w = int(w/m * self.max_image_size)
        image = image.resize((h, w)).convert("RGB")
        
        prompt = [
            {
                "role": "user", "content": [
                    {"type": "image", "image" : image}, 
                    {"type": "text", "text": f"{question} Also first provide your reasons between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put either real/fake) {SOLUTION_END}."}
                ]
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize = False,
            add_generation_prompt = True, # Must add assistant
        )
        return {"prompt": prompt, "image": image, "answer": label}
