import argparse
from pathlib import Path

from .dataset_builder import build_dataset
from .registry import get_persona
from .training_config import TrainingConfig


def _load_split_datasets(train_path: Path, val_path: Path):
    from datasets import load_dataset

    data_files = {"train": str(train_path), "validation": str(val_path)}
    return load_dataset("json", data_files=data_files)


def _formatting_func(example: dict) -> str:
    return (
        f"<|system|>\n{example['system']}\n"
        f"<|instruction|>\n{example['instruction']}\n"
        f"<|response|>\n{example['response']}"
    )


def _build_quantization_config(use_4bit: bool):
    if not use_4bit:
        return None

    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def train(config: TrainingConfig) -> Path:
    """Fine-tune a persona's base model with LoRA on its instruction dataset."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    get_persona(config.persona_id)  # validates persona_id exists

    training_data_dir = Path(__file__).resolve().parent.parent / "training_data"
    train_path = training_data_dir / f"{config.persona_id}_train.jsonl"
    val_path = training_data_dir / f"{config.persona_id}_val.jsonl"
    if not train_path.exists() or not val_path.exists():
        train_path, val_path = build_dataset(config.persona_id)

    dataset = _load_split_datasets(train_path, val_path)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _build_quantization_config(config.use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
    )

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_length=config.max_seq_length,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        formatting_func=_formatting_func,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a persona model with LoRA.")
    parser.add_argument("--persona", required=True, help="Persona id from the registry.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps (e.g. for a smoke test).")
    parser.add_argument("--use-4bit", action="store_true", help="Enable 4-bit QLoRA quantization.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override checkpoint output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = TrainingConfig(
        persona_id=args.persona,
        use_4bit=args.use_4bit,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    saved_to = train(cfg)
    print(f"Checkpoint saved to: {saved_to}")
