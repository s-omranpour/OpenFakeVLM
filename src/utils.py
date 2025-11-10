import torch

def normalize_answer(raw_answer: str) -> str:
    if 'real' in raw_answer:
        return "real"
    if 'fake' in raw_answer:
        return "fake"
    return "unknown"

def run_batch(
    model, tokenizer, batch,
    max_new_tokens = 5, 
    temperature = 0.7,
    top_p = 0.8,
    top_k = 20,
):
    gen_ids = model.generate(
        **batch, 
        max_new_tokens = max_new_tokens, 
        use_cache = True, 
        temperature = temperature, 
        top_p = top_p
    )
    gen_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, gen_ids)]
    output = tokenizer.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    predictions = list(map(normalize_answer, output))
    return predictions