import argparse
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.functional.text.rouge import rouge_score

from src.datasets import FakeClueSFTDataset, TestDataCollator
from src.utils import run_batch

torch.set_float32_matmul_precision('high')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a vision-language model (VLM) using Unsloth and TRL on the FakeClue dataset."
    )
    
    # Paths
    parser.add_argument("--model_name_path", type=str, default="unsloth/Qwen3-VL-2B-instruct", help="Model name or path")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset directory containing train/test splits.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints and logs.")

    # Flags
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--test", action="store_true", help="Run testing phase")
    

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer.")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs.")

    # Logging
    parser.add_argument("--project_name", type=str, default="OpenFakeVLM",
                        help="W&B project name for logging.")
    parser.add_argument("--log_steps", type=int, default=50,
                        help="Logging interval (in steps).")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation interval (in steps).")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Model saving interval (in steps).")

    return parser.parse_args()


def main(args):
    ## Load Model
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name_path,
        load_in_16bit=True,
        full_finetuning=True
    )

    ## Initialize Logger
    wandb.init(project=args.project_name, config=vars(args))

    if args.train:
        
        ## Load Dataset
        dataset = FakeClueSFTDataset(args.data_path, split='train', conversational=True)
        n = len(dataset)
        t = n // 10
        train_data, val_data = random_split(dataset, [n - t, t])
        
        ## Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
    
            args=SFTConfig(
                max_length=None,
                dataloader_num_workers=4,
                assistant_only_loss=True,
    
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.accumulation_steps,
    
                optim="adamw_torch_fused",
                weight_decay=args.weight_decay,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                max_steps=50,
                # num_train_epochs=args.num_epochs,
                learning_rate=args.lr,
    
                # Logging and evaluation
                logging_steps=args.log_steps,
                eval_steps=args.eval_steps,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=args.save_steps,
                save_total_limit=1,
    
                # Mixed precision and gradient clipping
                # bf16=True,
                max_grad_norm=0.5,
    
                output_dir=args.save_dir,
                report_to="wandb",
            ),
        )
        trainer.train()

    ## Evaluate
    if args.test:
        test_data = FakeClueSFTDataset(args.data_path, split='test', conversational=False)
        test_loader = DataLoader(
            test_data, 
            collate_fn=TestDataCollator(tokenizer), 
            batch_size=args.batch_size, num_workers=4, shuffle=False
        )
        
        FastVisionModel.for_inference(model)  # Enable inference mode
        predictions, labels = [], []
        outputs, ground_truths = [], []
    
        for (batch, gt, ans) in tqdm(test_loader):
            pr, raw_out = run_batch(model, tokenizer, batch.to('cuda'))
            predictions += pr
            labels += gt
            outputs += raw_out
            ground_truths += ans
    
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        rouge_l = rouge_score(outputs, ground_truths, use_stemmer=True)['rougeL_fmeasure'].item()
        
        wandb.log({'accuracy': acc})
        wandb.log({'f1': f1})
        wandb.log({'rouge-L': rouge_l})
        print('Accuracy:', acc, '  F1:', f1, 'ROUGE-L:', rouge_l)

    # model.save_pretrained(args.save_dir + "Qwen3-VL-2B-instruct-FakeClues-sft")  # Local saving
    # tokenizer.save_pretrained(args.save_dir + "Qwen3-VL-2B-instruct-FakeClues-sft")


if __name__ == "__main__":
    args = parse_args()
    main(args)