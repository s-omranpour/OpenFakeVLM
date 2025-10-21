# OpenFakeVLM


## Evaluation

 - SwinTransformer-V2 trained on OpenFake:
   Accuracy: 52.76%

    | Class        | Precision | Recall  | F1-score |
    |--------------|-----------|---------|----------|
    | AI-generated | 0.9511    | 0.2741  | 0.4256   |
    | Real         | 0.4321    | 0.9751  | 0.5988   |

 - SwinTransformer-V2 trained on OpenFake + DF40 + GenImage:
   Accuracy: 65.73%

    | Class        | Precision | Recall  | F1-score |
    |--------------|-----------|---------|----------|
    | AI-generated | 0.9805    | 0.4727  | 0.6379   |
    | Real         | 0.5137    | 0.9834  | 0.6749   |
