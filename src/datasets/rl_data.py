import pandas as pd
import numpy as np
import os
from PIL import Image
import torch


class FakeCluePromptDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train'):
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        json_path = os.path.join(data_dir, f'data_json/{split}.json')
        self.df = pd.read_json(json_path)
        self.df = self.df[(self.df.cate != 'doc') & (self.df.cate != 'satellite')]
        self.df.reset_index(inplace=True, drop=True)

        self.conversational = conversational
        self.prompt_only = prompt_only

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
        question = 'Does the image look real/fake?')
        
        prompt = [
            {
                "role": "user", "content": [
                    {"type": "image", "image" : image}, 
                    {"type": "text", "text": f"{question} Also first provide your reasons between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put either real/fake) {SOLUTION_END}."}
                ]
            }
        ]
        return {"prompt": prompt, "image": image, "answer": label}
