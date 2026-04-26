"""Shared configuration for the GRPO + QLoRA Phase 1 training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FALLBACK_ACTION = {
    "action_type": "exclude",
    "final_decision_reason": "fallback_invalid_output",
}


@dataclass(slots=True)
class Phase1Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    env_url: str = "http://localhost:7860"
    task_id: str = "task3"
    seed_start: int = 100
    num_episodes: int = 8
    output_dir: str = "artifacts/phase1_grpo"
    max_steps: int = 15
    num_generations: int = 3
    generation_batch_size: int = 6
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-6
    grpo_epsilon: float = 0.05
    seed: int = 42
    max_actions: int = 14
    max_new_tokens: int = 32
    local_debug_mode: bool = False
    collect_debug_rollouts: bool = True
    sft_warmstart_epochs: int = 1
    sft_learning_rate: float = 5e-6

    def validate(self) -> None:
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                "generation_batch_size must be divisible by num_generations for stable GRPO batching"
            )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
}


QLORA_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
}
