# OpenFakeVLM

## Template 
Prompt:
```
Does the image look real/fake?
Also first provide your reasons between <REASONING> and </REASONING> and then your final answer between <SOLUTION> and (put either real/fake) </SOLUTION>.
```

## Results

### **Qwen3-VL-2B-instruct (Zero-shot)**
- **Accuracy:** 56.67. 
- **F1:** 22.66  
- **ROUGE-L:** 19.53


### **Qwen3-VL-2B-instruct (SFT for 2 epochs)**
- **Accuracy:** 90.88  
- **F1:** 61.06  
- **ROUGE-L:** 47.45

### **Qwen3-VL-8B-instruct (SFT for 2 epochs)**
- **Accuracy:** 82.88  
- **F1:** 59.25   
- **ROUGE-L:** 42.32

## Todos
- [ ] Add support for LOKI and DD-VQA datasets    
- [x] Train and evaluate Qwen3-VL-8B
 
