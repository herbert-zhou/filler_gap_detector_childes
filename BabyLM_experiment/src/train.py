from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import sample, seed
from pathlib import Path
import yaml
import argparse
import json 
from transformers import set_seed
from babylm_dataset import BabylmDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config" / "llama-360M.yaml"), help="Configuration file path")
parser.add_argument("--lr", type=float, default=None, help="Learning rate")
parser.add_argument("--filter", type=str, default=None, help="Filter to use")
parser.add_argument("--include", type=str, default="par", help="Include all or only parent, use all for including all and par for only parent")
parser.add_argument("--size", type=str, default="10M", help="Size of dataset")
parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")

parser.add_argument("--control", action="store_true", help="Use control")

# Balanced-membership dataset support
parser.add_argument("--data_root", type=str, default=None,
                    help="Direct path to data root (e.g. data/filtered_balanced/par/rep_0/embq+matq). "
                         "Bypasses --filter/--include path construction. Must contain train_{size}/ and test/ subdirs.")
parser.add_argument("--model_output_root", type=str, default=None,
                    help="Direct path to model output directory (used with --data_root). "
                         "If not set with --data_root, derived from data_root path.")


args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


# Override config parameters if provided as command-line arguments
if args.lr:
    config['training']['lr'] = args.lr
if args.weight_decay is not None:
    config['training']['weight_decay'] = args.weight_decay

model_name = Path(args.config).name[:-len(".yaml")]
print("model -- ", model_name)

SEQ_LENGTH = int(config['data']['seq_length'])

BASE_DATA_PATH = PROJECT_ROOT / "data" / "filtered"
BASE_OUTPUT_DIR = PROJECT_ROOT / "models"

tokenizer_path = PROJECT_ROOT / "models" / "gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path), truncation=True, max_length=SEQ_LENGTH)
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

if args.data_root:
    # Balanced-membership dataset mode: use direct paths
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    balanced_train_path = data_root / "train_balanced"
    legacy_train_path = data_root / f"train_{args.size}"
    train_path = balanced_train_path if balanced_train_path.exists() else legacy_train_path
    eval_path = data_root / "test"
elif args.control:
    train_path = Path(BASE_DATA_PATH, args.include, args.filter + "_control", f"train_{args.size}")
    eval_path = Path(BASE_DATA_PATH, args.include, args.filter + "_control", f"test")
else:
    train_path = Path(BASE_DATA_PATH, args.include, args.filter, f"train_{args.size}")
    # Evaluated on control!!!!
    eval_path = Path(BASE_DATA_PATH, args.include, args.filter, f"test")

train_dataset = BabylmDataset(train_path, SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(eval_path, SEQ_LENGTH, tokenizer=tokenizer, offset=0)

seed(2023) # we fix the same subset for all models
print("LENGTH EVAL", len(full_eval_dataset), "\n")
eval_indices = sample(range(len(full_eval_dataset)), int(config['data']['eval_samples']))
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Dynamic Model Configuration
if config['model']['type'] == "Llama":
    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2*tokenizer.model_max_length,
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        num_hidden_layers=config['model']['n_layer'],
        num_attention_heads=config['model']['n_head'],
        tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = LlamaForCausalLM(model_config)
elif config['model']['type'] == "GPT2":
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=2*tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        resid_pdrop = config['model']['resid_pdrop'],
        embd_pdrop = config['model']['embd_pdrop'],
        attn_pdrop = config['model']['attn_pdrop'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = GPT2LMHeadModel(model_config)
elif config['model']['type'] == "GPTJ":
    model_config = GPTJConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=2*tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        resid_pdrop = config['model']['resid_pdrop'],
        embd_pdrop = config['model']['embd_pdrop'],
        attn_pdrop = config['model']['attn_pdrop'],
        tie_word_embeddings=config['model']['tie_word_embeddings'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = GPTJForCausalLM(model_config)

model.to(device)
print(f'model parameters = {model.num_parameters()}')

# Create unique output directory with hyperparameters
lr_str = f"lr{float(config['training']['lr']):.0e}".replace('-', '')
wd_str = f"wd{float(config['training'].get('weight_decay', 0.0)):.0e}".replace('-', '')
hparam_suffix = f"{lr_str}_{wd_str}"

if args.data_root:
    # Balanced-membership mode: output dir derived from data_root or explicit flag
    if args.model_output_root:
        output_dir = Path(args.model_output_root) / f"{model_name}-{args.size}" / hparam_suffix
    else:
        # Derive from data_root: data/filtered_balanced/par/rep_0/embq+matq -> models_balanced/par/rep_0/embq+matq
        data_root = Path(args.data_root)
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        # Find the part after 'filtered_balanced'
        parts = data_root.parts
        try:
            idx = parts.index("filtered_balanced")
            rel = Path(*parts[idx + 1:])  # e.g. par/rep_0/embq+matq
        except ValueError:
            rel = data_root.name
        output_dir = PROJECT_ROOT / "models_balanced" / rel / f"{model_name}-{args.size}" / hparam_suffix
elif args.control:
    output_dir = Path(BASE_OUTPUT_DIR, args.include, args.filter, f"{model_name}-{args.size}-control", hparam_suffix)
else:
    output_dir = Path(BASE_OUTPUT_DIR, args.include, args.filter, f"{model_name}-{args.size}", hparam_suffix)

accumulation_steps = int(config['training']['gradient_accumulation_steps'])
per_device_bsz = int(config['training']['batch_size']) // accumulation_steps

if __name__ == "__main__":

    # if config['logging']['wandb']:
    #     import wandb
    #     wandb.login()
    #     wandb.init(project= config['logging']['project'], name=config['model']['name'], config=config)
    seed = args.seed
    output_dir_plus_seed = Path(output_dir, f"seed_{seed}")
    final_dir = Path(output_dir_plus_seed, "final")

    print("Outputting to ", output_dir_plus_seed)


    training_args = TrainingArguments(
        output_dir=output_dir_plus_seed,
        # overwrite_output_dir=True,
        metric_for_best_model = "eval_loss",
        eval_strategy = "epoch",
        save_strategy = "no",  # Disable checkpoint saving to save disk space
        num_train_epochs=int(config['training']['num_epochs']),
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        warmup_steps=int(config['training']['warmup_steps']), 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        weight_decay=float(config['training'].get('weight_decay', 0.0)),
        logging_steps=20,
        fp16=config['training']['fp16'],
        # load_best_model_at_end=True,
        torch_compile = config['training'].get('torch_compile', False),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    set_seed(seed)
    trainer.train()
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    log_history = trainer.state.log_history

    # Filter for evaluation results
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    # Save to a JSON file
    with open(Path(output_dir_plus_seed, f"trainer_eval_logs_{model_name}.json"), 'w') as f:
        json.dump(eval_logs, f, indent=4)