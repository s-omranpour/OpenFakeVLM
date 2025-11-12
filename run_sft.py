import argparse
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.functional.text.rouge import rouge_score


from src.datasets.sft_data import FakeClueChatDataset
from src.datasets.collators import TestDataCollator
from src.utils import run_batch

from accelerate import Accelerator


# torch.set_float32_matmul_precision('high')


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
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Logging interval (in steps).")
    parser.add_argument("--eval_steps", type=int, default=400,
                        help="Evaluation interval (in steps).")
    parser.add_argument("--save_steps", type=int, default=400,
                        help="Model saving interval (in steps).")

    return parser.parse_args()


def main(args):
    ## Load Model
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name_path,
        # load_in_16bit=True,
        full_finetuning=True,
        float32_mixed_precision = True
    )

    if args.train:
        wandb.init(project=args.project_name, config=vars(args))
        
        ## Load Dataset
        dataset = FakeClueChatDataset(args.data_path, split='train', conversational=True)
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
                max_length=2048,
                dataloader_num_workers=4,
                assistant_only_loss=True,

                ddp_find_unused_parameters=False,
    
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.accumulation_steps,
    
                optim="adamw_torch_fused",
                weight_decay=args.weight_decay,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                num_train_epochs=args.num_epochs,
                learning_rate=args.lr,
    
                # Logging and evaluation
                logging_steps=args.log_steps,
                eval_steps=args.eval_steps,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=args.save_steps,
                save_total_limit=1,
                load_best_model_at_end = True,
                metric_for_best_model = "eval_loss", # metric we want to early stop on
                greater_is_better = False,
    
                # Mixed precision and gradient clipping
                max_grad_norm=0.5,
    
                output_dir=args.save_dir,
                report_to="wandb",
            ),
        )
        trainer.train()

    ## Evaluate
    if args.test:
        # Initialize accelerator
        accelerator = Accelerator()
        device = accelerator.device

        FastVisionModel.for_inference(model)  # Enable inference mode
        
        test_data = FakeClueChatDataset(
            args.data_path, 
            split='test',
            conversational=False#, max_num_samples=100
        )
        test_loader = DataLoader(
            test_data, 
            collate_fn=TestDataCollator(tokenizer), 
            batch_size=8, num_workers=4, shuffle=False
        )
        model, test_loader = accelerator.prepare(model, test_loader)
        model = accelerator.unwrap_model(model)
        
        reasons = []
        predictions = []
        gt_reasons = []
        gt_labels = []
        for (batch, gt_r, gt_l) in tqdm(test_loader):
            pred_reasons, pred_answers, _ = run_batch(
                model, tokenizer, batch.to(device), max_new_tokens=512, temperature=0.1
            )
            reasons += pred_reasons
            predictions += pred_answers
            gt_reasons += gt_r
            gt_labels += gt_l
            
        
        # Gather predictions from all processes
        accelerator.wait_for_everyone()
        reasons = accelerator.gather_for_metrics(reasons, use_gather_object=True)
        predictions = accelerator.gather_for_metrics(predictions, use_gather_object=True)
        gt_reasons = accelerator.gather_for_metrics(gt_reasons, use_gather_object=True)
        gt_labels = accelerator.gather_for_metrics(gt_labels, use_gather_object=True)

        if accelerator.is_main_process:
            model_predictions_df = pd.DataFrame({
                "reasons": reasons,
                "predictions": predictions,
                "gt_reasons" : gt_reasons,
                "gt_labels" : gt_labels
            }).drop_duplicates(['gt_reasons'])
            acc = accuracy_score(gt_labels, predictions) * 100
            f1 = f1_score(gt_labels, predictions, average='macro').item() * 100
            rouge_l = rouge_score(
                reasons, 
                gt_reasons, 
                use_stemmer=True
            )['rougeL_fmeasure'].item() * 100
            print('Data size:', len(test_data), ' preds size:', model_predictions_df.shape[0])
            print('Accuracy:', acc, '  F1:', f1, 'ROUGE-L:', rouge_l)
            model_predictions_df.to_csv(
                args.save_dir +\
                f"model_predictions-acc={round(acc, 2)}-f1={round(f1, 2)}-rougeL={round(rouge_l, 2)}.csv",
                index=False
            )
        

    # model.save_pretrained(args.save_dir)  # Local saving
    # tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)