REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

class TestDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = []
        texts = []
        for sample in batch:
            images += [sample['image']]
            
            messages = [
                {
                    "role": "user", "content": [
                        {"type": "image"}, 
                        {"type": "text", "text": f"{sample['question']} Also first provide your reasons between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put either real/fake) {SOLUTION_END}."}
                    ]
                }
            ]
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
            texts += [input_text]
    
        inputs = self.tokenizer(
            images,
            texts,
            add_special_tokens = False,
            padding=True,
            return_tensors = "pt",
        )
        return inputs