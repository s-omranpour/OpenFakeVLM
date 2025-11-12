import re
import torch

REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def process_answer(raw_answer: str):
    reasoning_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
    

    reasoning = re.findall(reasoning_pattern, raw_answer, re.DOTALL)
    if len(reasoning) > 0:
        reasoning = reasoning[0].strip()
    else:
        reasoning = ''
        
    answer = re.findall(answer_pattern, raw_answer, re.DOTALL)
    if len(answer) > 0:
        answer = answer[0].strip()
    else:
        answer = ''
    return reasoning, answer
    

def run_batch(
    model, tokenizer, batch,
    max_new_tokens = 512, 
    temperature = 0.7,
    top_p = 0.8,
    top_k = 20,
):
    gen_ids = model.generate(
        **batch, 
        max_new_tokens = max_new_tokens, 
        use_cache = True, 
        temperature = temperature, 
        top_p = top_p,
        top_k = top_k
    )
    gen_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, gen_ids)]
    raw_output = tokenizer.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    processed_output = list(map(process_answer, raw_output))
    reasons = [po[0] for po in processed_output]
    answers = [po[1] for po in processed_output]
    return reasons, answers, raw_output



# Reward functions
def formatting_reward_func(completions, **kwargs):
    reasoning_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    scores = []
    for completion in completions:
        score = 0
        reasoning_matches = re.findall(reasoning_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(reasoning_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0
        scores.append(score)
    return scores


def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    return [
        3.0 if len(r)==1 and a == r[0].replace('\n','') else 0.0
        for r, a in zip(responses, answer)
    ]