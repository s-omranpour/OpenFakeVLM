import pandas as pd
import numpy as np
import os
from PIL import Image
import torch


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
ANSWER_START = "<ANSWER>"
ANSWER_END = "</ANSWER>"

class FakeClueDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, 
        split='train', 
        conversational=True, 
        add_reasoning=True,
        max_image_size=512, 
        max_num_samples=None
    ):
        super().__init__()
        self.conversational = conversational
        self.max_image_size = max_image_size
        self.add_reasoning = add_reasoning

        self.prompt = f"Answer the question 'Does the image look real/fake?' between {ANSWER_START} and {ANSWER_END}."
        if self.add_reasoning:
            self.prompt += f'Then, provide your reasons between {REASONING_START} and {REASONING_END}.'
             
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
        

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        samp = self.df.loc[idx]
        label = samp['label']
        reason = samp['answer']
        
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
        
        if self.conversational:
            response = f"{ANSWER_START}{label}{ANSWER_END}"
            if self.add_reasoning:
                response += f"\n{REASONING_START}\n{reason}\n{REASONING_END}"

            messages = [
                {
                    "role": "user", "content": [
                        {"type": "image", "image" : image}, 
                        {"type": "text", "text": self.prompt}
                    ]
                },
                {
                    "role": "assistant", "content": [
                        {"type": "text", "text": response}
                    ]
                }
            ]
            return {'messages' : messages}
        return {'image' : image, 'prompt' : self.prompt, 'reason' : reason, 'label' : label}




class TestDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = []
        texts = []
        gt_reasons = []
        gt_labels = []
        for sample in batch:
            images += [sample['image']]
            
            messages = [
                {
                    "role": "user", "content": [
                        {"type": "image"}, 
                        {"type": "text", "text": sample['prompt']}
                    ]
                }
            ]
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
            texts += [input_text]
            gt_reasons += [sample['reason']]
            gt_labels += [sample['label']]
    
        inputs = self.tokenizer(
            images,
            texts,
            add_special_tokens = False,
            padding=True,
            return_tensors = "pt",
        )
        return inputs, (gt_labels, gt_reasons)