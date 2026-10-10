from dataclasses import dataclass, field
from pathlib import Path

_CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


@dataclass
class TrainingConfig:
    persona_id: str
    # Instruction-tuned so it already answers the user; LoRA only layers the persona's voice on top.
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    use_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    num_train_epochs: int = 1
    max_steps: int | None = None  # overrides num_train_epochs when set (e.g. smoke tests)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_seq_length: int = 512
    output_dir: Path | None = None  # defaults to checkpoints/{persona_id}

    def resolved_output_dir(self) -> Path:
        return self.output_dir if self.output_dir is not None else _CHECKPOINTS_DIR / self.persona_id
