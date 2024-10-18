import argparse
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import os
import torch
import torch.nn as nn
from transformers import TrainerCallback
from contextlib import contextmanager
import sys
import warnings
from prettytable import PrettyTable  # Import PrettyTable

# Suppress wandb
os.environ["WANDB_DISABLED"] = "true"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
GLUE_TASKS = ["stsb", "cola", "mrpc", "rte", "wnli"]
MODEL_CHECKPOINT = "bert-base-uncased"

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with LoRA parameters')
    parser.add_argument('--rank', type=int, required=True, help='Rank value for LoRA')
    parser.add_argument('--a', type=int, required=True, help='A value for LoRA')
    parser.add_argument('--b', type=int, required=True, help='B value for LoRA')
    return parser.parse_args()

# Utility functions
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def preprocess_function(examples, task_name, tokenizer):
    if task_name in ['cola', 'sst2']:
        return tokenizer(examples['sentence'], truncation=True)
    elif task_name == 'stsb':
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)
    elif task_name == 'qqp':
        return tokenizer(examples['question1'], examples['question2'], truncation=True)
    elif task_name == 'qnli':
        return tokenizer(examples['question'], examples['sentence'], truncation=True)
    elif task_name == 'mnli':
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True)
    else:
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

def compute_metrics(task_name, metric, eval_pred):
    predictions, labels = eval_pred
    if task_name == 'stsb':
        predictions = np.squeeze(predictions)
        task_metric = metric.compute(predictions=predictions, references=labels)
        return {**task_metric}
    else:
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        task_metric = metric.compute(predictions=predictions, references=labels)
        return {**task_metric, "accuracy": accuracy}

class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            print(f"\nMetrics after epoch {state.epoch}: {metrics}\n")

def merge_l(model, lora_dim, lora_a, lora_b):
    # Import LoRA implementation
    import lidb_but_train_is_small_randn as lora
    
    target_names = []
    for name, module in model.named_modules():
        if "dense" in name:
            target_names.append(name)
    
    for name in target_names:
        name_struct = name.split(".")
        module_list = [model]
        for struct in name_struct:
            module_list.append(getattr(module_list[-1], struct))
        lora_layer = lora.Linear(
            module_list[-1].in_features, 
            module_list[-1].out_features, 
            r=lora_dim,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_alpha=16  
        ).to(device)
        
        lora_layer.weight.data = module_list[-1].weight.data
        if module_list[-1].bias is not None:
            lora_layer.bias.data = module_list[-1].bias.data
        
        module_list[-2].__setattr__(name_struct[-1], lora_layer)
    
    # Set parameter gradients
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = "train" in name
        elif "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model

def main():
    # Parse arguments
    args = parse_args()
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, 
                            message="Some weights of .* were not initialized from the model checkpoint.*")
    
    # Set the logging level to suppress progress bars
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    # Suppress warnings
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=1,
        logging_steps=500,
        save_steps=1000,
        do_train=True,
        do_eval=True,
        report_to="none"
    )
    
    results_train = []

    # Create a table for results
    results_table = PrettyTable()
    results_table.field_names = ["Task", "Accuracy"]

    for task in GLUE_TASKS:
        print(f"\nFine-tuning on {task.upper()} task with rank={args.rank}, a={args.a}, b={args.b}...")
        
        # Load dataset and metric
        print(f"Loading dataset and metric for task: {task.upper()}")
        with suppress_output():
            dataset = load_dataset("glue", task)
            metric = evaluate.load("glue", task)
            num_labels = 1 if task == "stsb" else dataset['train'].features['label'].num_classes
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)
            encoded_dataset = dataset.map(
                lambda x: preprocess_function(x, task, tokenizer), 
                batched=True
            )
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Apply LoRA
        print(f"Applying merge with rank={args.rank}, a={args.a}, b={args.b}...")
        model = merge_l(model, args.rank, args.a, args.b)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(task, metric, eval_pred),
            #callbacks=[LogMetricsCallback()]
        )
        
        # Train and evaluate
        trainer.train()
        result = trainer.evaluate()
        
        # Store results
        acc = result.get('eval_accuracy', result.get('eval_pearson', 0))
        results_train.append(acc)
        
        # Add results to the table
        results_table.add_row([task.upper(), acc])

        print(f"Results for {task.upper()}: {result}\n")
    
    print(f"Final results for rank={args.rank}, a={args.a}, b={args.b}: {results_train}")
    
    # Print the final results table
    print(results_table)
    
    return results_train

if __name__ == "__main__":
    main()
