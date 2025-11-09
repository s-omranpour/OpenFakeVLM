import torch

def normalize_answer(raw_answer: str) -> str:
    if 'real' in raw_answer:
        return "real"
    if 'fake' in raw_answer:
        return "fake"
    return "unknown"

def run_batch(model, tokenizer, batch):
    images = []
    texts = []
    true_labels = []
    for sample in batch:
        images += [sample['image']]
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample['question']}]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        texts += [input_text]
        true_labels += [sample['label']]

    inputs = tokenizer(
        images,
        texts,
        add_special_tokens = False,
        padding=True,
        return_tensors = "pt",
    ).to("cuda")

    gen_ids = model.generate(**inputs, max_new_tokens = 5, use_cache = True, temperature = 0.7, top_p = 0.8)
    gen_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    output = tokenizer.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    predictions = list(map(normalize_answer, output))
    return predictions, true_labels