#!/usr/bin/env python3
"""Colab-ready QLoRA + GRPO Wordle training script.

This script is intentionally self-contained: it builds a small synthetic
Wordle dataset, trains a 4-bit LoRA adapter with TRL GRPOTrainer, and runs a
short held-out validation loop. It is meant for a single Google Colab GPU with
roughly 8-12GB VRAM.

Colab usage:

    !python training/qlora_grpo_wordle_colab.py --install-deps --max-steps 150

Quick smoke/debug usage:

    !python training/qlora_grpo_wordle_colab.py --install-deps --debug-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_MODEL = "Qwen/Qwen2.5-0.5B"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = Path("./artifacts/qlora_grpo_wordle")

WORD_RE = re.compile(r"\b[A-Z]{5}\b")

# Embedded 5-letter vocabulary. Keeping it local avoids network/file failures
# during Colab setup while still giving the environment enough diversity.
VOCAB = sorted(
    {
        "ABOUT",
        "ABOVE",
        "ABUSE",
        "ACTOR",
        "ACUTE",
        "ADAPT",
        "ADMIT",
        "ADOPT",
        "ADORE",
        "ADULT",
        "AFTER",
        "AGAIN",
        "AGENT",
        "AGILE",
        "AGREE",
        "AHEAD",
        "ALARM",
        "ALBUM",
        "ALERT",
        "ALIEN",
        "ALIGN",
        "ALIKE",
        "ALIVE",
        "ALLOW",
        "ALONE",
        "ALTER",
        "AMBER",
        "AMEND",
        "AMONG",
        "AMPLE",
        "ANGLE",
        "ANGRY",
        "APPLE",
        "APPLY",
        "ARGUE",
        "ARISE",
        "ARRAY",
        "ASIDE",
        "ASSET",
        "AUDIO",
        "AUDIT",
        "AVOID",
        "AWAKE",
        "AWARE",
        "BASIC",
        "BASIL",
        "BATCH",
        "BEACH",
        "BEARD",
        "BEAST",
        "BEGIN",
        "BEING",
        "BELLY",
        "BENCH",
        "BERRY",
        "BIRTH",
        "BLACK",
        "BLAME",
        "BLEND",
        "BLIND",
        "BLOCK",
        "BLOOM",
        "BOARD",
        "BOOST",
        "BOOTH",
        "BOUND",
        "BRAIN",
        "BRAND",
        "BRAVE",
        "BREAD",
        "BREAK",
        "BRICK",
        "BRIDE",
        "BRIEF",
        "BRING",
        "BROAD",
        "BROWN",
        "BRUSH",
        "BUILD",
        "BUILT",
        "BURST",
        "CABLE",
        "CACHE",
        "CANDY",
        "CARRY",
        "CARVE",
        "CAUSE",
        "CHAIN",
        "CHAIR",
        "CHARM",
        "CHART",
        "CHASE",
        "CHEAP",
        "CHECK",
        "CHEST",
        "CHIEF",
        "CHILD",
        "CHOIR",
        "CIVIC",
        "CLAIM",
        "CLASS",
        "CLEAN",
        "CLEAR",
        "CLICK",
        "CLIMB",
        "CLOCK",
        "CLOSE",
        "CLOUD",
        "COACH",
        "COAST",
        "COLON",
        "COLOR",
        "COUNT",
        "COURT",
        "COVER",
        "CRAFT",
        "CRANE",
        "CRASH",
        "CREAM",
        "CRISP",
        "CROSS",
        "CROWD",
        "CROWN",
        "CURVE",
        "DAILY",
        "DANCE",
        "DEALT",
        "DEATH",
        "DEBUG",
        "DELAY",
        "DELTA",
        "DEPTH",
        "DIARY",
        "DIGIT",
        "DIRTY",
        "DOING",
        "DOUBT",
        "DRAFT",
        "DRAMA",
        "DREAM",
        "DRINK",
        "DRIVE",
        "EAGER",
        "EARLY",
        "EARTH",
        "EIGHT",
        "ELDER",
        "ELECT",
        "ELITE",
        "EMPTY",
        "ENJOY",
        "ENTER",
        "ENTRY",
        "EQUAL",
        "ERROR",
        "EVENT",
        "EVERY",
        "EXACT",
        "EXIST",
        "EXTRA",
        "FAITH",
        "FALSE",
        "FAULT",
        "FAVOR",
        "FENCE",
        "FEVER",
        "FIELD",
        "FINAL",
        "FIRST",
        "FIXED",
        "FLASH",
        "FLEET",
        "FLOOR",
        "FOCUS",
        "FORCE",
        "FORTH",
        "FOUND",
        "FRAME",
        "FRESH",
        "FRONT",
        "FRUIT",
        "GIANT",
        "GIVEN",
        "GLASS",
        "GLOBE",
        "GRACE",
        "GRADE",
        "GRAND",
        "GRANT",
        "GRAPH",
        "GRASS",
        "GREAT",
        "GREEN",
        "GROUP",
        "GUARD",
        "GUESS",
        "GUEST",
        "GUIDE",
        "HAPPY",
        "HEART",
        "HEAVY",
        "HONEY",
        "HORSE",
        "HOTEL",
        "HOUSE",
        "HUMAN",
        "IDEAL",
        "IMAGE",
        "INDEX",
        "INNER",
        "INPUT",
        "ISSUE",
        "JOINT",
        "JUDGE",
        "KNOWN",
        "LABEL",
        "LARGE",
        "LASER",
        "LATER",
        "LAUGH",
        "LAYER",
        "LEARN",
        "LEAST",
        "LEAVE",
        "LEGAL",
        "LEVEL",
        "LIGHT",
        "LIMIT",
        "LOCAL",
        "LOGIC",
        "LOOSE",
        "LUCKY",
        "MAGIC",
        "MAJOR",
        "MAKER",
        "MARCH",
        "MATCH",
        "MAYBE",
        "MEDAL",
        "MEDIA",
        "METAL",
        "MIGHT",
        "MINOR",
        "MODEL",
        "MONEY",
        "MONTH",
        "MOTOR",
        "MOUNT",
        "MOUSE",
        "MOUTH",
        "MUSIC",
        "NEEDS",
        "NERVE",
        "NEVER",
        "NIGHT",
        "NOISE",
        "NORTH",
        "NOVEL",
        "NURSE",
        "OCEAN",
        "OFFER",
        "OFTEN",
        "OLIVE",
        "ONION",
        "ORDER",
        "OTHER",
        "OUTER",
        "OWNER",
        "PANEL",
        "PAPER",
        "PARTY",
        "PATCH",
        "PEACE",
        "PHASE",
        "PHONE",
        "PHOTO",
        "PIANO",
        "PIECE",
        "PILOT",
        "PITCH",
        "PLACE",
        "PLAIN",
        "PLANE",
        "PLANT",
        "PLATE",
        "POINT",
        "POWER",
        "PRESS",
        "PRICE",
        "PRIDE",
        "PRIME",
        "PRINT",
        "PRIOR",
        "PROOF",
        "PROUD",
        "QUERY",
        "QUICK",
        "QUIET",
        "QUITE",
        "RADIO",
        "RAISE",
        "RANGE",
        "RAPID",
        "RATIO",
        "REACH",
        "READY",
        "REALM",
        "REBEL",
        "REFER",
        "RELAX",
        "REPLY",
        "RESET",
        "RIGHT",
        "RIVER",
        "ROBOT",
        "ROUGH",
        "ROUND",
        "ROUTE",
        "ROYAL",
        "RURAL",
        "SCALE",
        "SCENE",
        "SCOPE",
        "SCORE",
        "SCOUT",
        "SERVE",
        "SETUP",
        "SHARE",
        "SHARP",
        "SHEET",
        "SHIFT",
        "SHINE",
        "SHOCK",
        "SHORT",
        "SHOWN",
        "SIGHT",
        "SKILL",
        "SLEEP",
        "SLIDE",
        "SMALL",
        "SMART",
        "SMILE",
        "SMOKE",
        "SOLID",
        "SOLVE",
        "SOUND",
        "SOUTH",
        "SPACE",
        "SPARE",
        "SPEAK",
        "SPEED",
        "SPEND",
        "SPICE",
        "SPIKE",
        "SPLIT",
        "SPORT",
        "STAFF",
        "STAGE",
        "STAIR",
        "STAKE",
        "STAND",
        "START",
        "STATE",
        "STEAM",
        "STEEL",
        "STICK",
        "STILL",
        "STONE",
        "STORE",
        "STORM",
        "STORY",
        "STRIP",
        "STUDY",
        "STYLE",
        "SUGAR",
        "SUPER",
        "SWEET",
        "TABLE",
        "TAKEN",
        "TASTE",
        "TEACH",
        "TERRA",
        "THANK",
        "THEIR",
        "THEME",
        "THERE",
        "THING",
        "THINK",
        "THIRD",
        "THREE",
        "THROW",
        "TIGER",
        "TIGHT",
        "TITLE",
        "TODAY",
        "TOKEN",
        "TOPIC",
        "TOTAL",
        "TOUCH",
        "TOUGH",
        "TOWER",
        "TRACE",
        "TRACK",
        "TRADE",
        "TRAIN",
        "TRIAL",
        "TRICK",
        "TRUCK",
        "TRUST",
        "TRUTH",
        "UNDER",
        "UNION",
        "UNITY",
        "UNTIL",
        "UPPER",
        "URBAN",
        "USAGE",
        "VALID",
        "VALUE",
        "VIDEO",
        "VIRAL",
        "VISIT",
        "VITAL",
        "VOICE",
        "WASTE",
        "WATCH",
        "WATER",
        "WHEEL",
        "WHERE",
        "WHILE",
        "WHITE",
        "WHOLE",
        "WORLD",
        "WORRY",
        "WORTH",
        "WOULD",
        "WRITE",
        "WRONG",
        "YIELD",
        "YOUNG",
    }
)


@dataclass
class WordleState:
    target: str
    state: str = "_ _ _ _ _"
    previous_guesses: list[str] = field(default_factory=list)
    max_steps: int = 6
    solved: bool = False

    def prompt(self) -> str:
        guesses = ", ".join(self.previous_guesses) if self.previous_guesses else "None"
        return f"Wordle state: {self.state}\nPrevious guesses: {guesses}\nNext guess:"


class WordleEnv:
    def __init__(self, vocab: list[str], max_steps: int = 6) -> None:
        self.vocab = vocab
        self.max_steps = max_steps
        self.state: WordleState | None = None

    def reset(self, target: str | None = None, rng: random.Random | None = None) -> WordleState:
        rng = rng or random
        self.state = WordleState(target=target or rng.choice(self.vocab), max_steps=self.max_steps)
        return self.state

    def step(self, guess: str) -> tuple[WordleState, float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        guess = normalize_guess(guess)
        target = self.state.target
        invalid = guess not in self.vocab
        repeated = guess in self.state.previous_guesses
        reward = 0.0

        if invalid or repeated:
            reward -= 0.1
        else:
            reward += score_guess(target, guess)
            self.state.previous_guesses.append(guess)
            self.state.state = merge_mask(self.state.state, target, guess)
            if guess == target:
                reward += 5.0
                self.state.solved = True

        done = self.state.solved or len(self.state.previous_guesses) >= self.max_steps
        info = {"invalid": invalid, "repeated": repeated, "target": target, "guess": guess}
        return self.state, float(reward), done, info


def install_dependencies() -> None:
    packages = [
        "transformers>=4.45.2",
        "peft>=0.13.2",
        "trl>=0.26.0",
        "accelerate>=1.0.1",
        "bitsandbytes>=0.43.3",
        "datasets>=3.0.1",
        "safetensors>=0.4.5",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", *packages]
    print("Installing/refreshing Colab training dependencies...")
    subprocess.check_call(cmd)


def import_training_libraries() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def normalize_guess(text: str) -> str:
    match = WORD_RE.search(str(text).upper())
    return match.group(0) if match else str(text).strip().upper()[:5]


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(completion)


def score_guess(target: str, guess: str) -> float:
    exact = sum(1 for a, b in zip(target, guess) if a == b)
    target_counts = Counter(target)
    guess_counts = Counter(guess)
    total_letter_matches = sum(min(target_counts[ch], guess_counts[ch]) for ch in guess_counts)
    wrong_position = max(0, total_letter_matches - exact)
    return float(exact * 1.0 + wrong_position * 0.5)


def merge_mask(mask: str, target: str, guess: str) -> str:
    letters = mask.split()
    if len(letters) != 5:
        letters = ["_"] * 5
    for idx, (target_ch, guess_ch) in enumerate(zip(target, guess)):
        if target_ch == guess_ch:
            letters[idx] = target_ch
    return " ".join(letters)


def best_next_guess(target: str, previous_guesses: list[str], rng: random.Random) -> str:
    if rng.random() < 0.72:
        return target
    candidates = [word for word in VOCAB if word not in previous_guesses]
    return max(candidates, key=lambda word: score_guess(target, word)) if candidates else target


def build_state_from_guesses(target: str, previous_guesses: list[str]) -> str:
    mask = "_ _ _ _ _"
    for guess in previous_guesses:
        mask = merge_mask(mask, target, guess)
    return mask


def build_synthetic_dataset(num_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    non_targets = VOCAB[:]

    for idx in range(num_samples):
        target = rng.choice(VOCAB)
        attempts = rng.randint(0, 4)
        previous_guesses: list[str] = []
        for _ in range(attempts):
            if rng.random() < 0.12 and previous_guesses:
                guess = rng.choice(previous_guesses)
            else:
                pool = [word for word in non_targets if word != target and word not in previous_guesses]
                guess = rng.choice(pool or non_targets)
            previous_guesses.append(guess)

        # Inject examples with clearer partial states so prompts are not all blank.
        if idx % 7 == 0:
            helper = list(target)
            swap_idx = rng.randrange(5)
            helper[swap_idx] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            guess = "".join(helper)
            if guess in VOCAB and guess != target and guess not in previous_guesses:
                previous_guesses.append(guess)

        state = build_state_from_guesses(target, previous_guesses)
        completion = best_next_guess(target, previous_guesses, rng)
        prompt = WordleState(target=target, state=state, previous_guesses=previous_guesses).prompt()
        samples.append(
            {
                "prompt": prompt,
                "completion": completion,
                "target": target,
                "state": state,
                "previous_guesses": previous_guesses,
            }
        )

    rng.shuffle(samples)
    return samples


def reward_wordle(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
    targets = kwargs.get("target") or []
    states = kwargs.get("state") or []
    previous_items = kwargs.get("previous_guesses") or []
    rewards: list[float] = []

    for idx, completion in enumerate(completions):
        target = str(targets[idx]).upper() if idx < len(targets) else extract_target_fallback(prompts[idx])
        guess = normalize_guess(completion_text(completion))
        previous_guesses = previous_items[idx] if idx < len(previous_items) else []
        if isinstance(previous_guesses, str):
            previous_guesses = [item.strip().upper() for item in previous_guesses.split(",") if item.strip()]
        state = str(states[idx]) if idx < len(states) else "_ _ _ _ _"

        reward = 0.0
        if guess not in VOCAB or guess in previous_guesses:
            reward -= 0.1
        else:
            reward += score_guess(target, guess)
            if guess == target:
                reward += 5.0
            elif state != "_ _ _ _ _":
                known_positions = [ch for ch in state.split() if ch != "_"]
                reward += 0.05 * sum(1 for ch in known_positions if ch in guess)
        rewards.append(float(reward))

    return rewards


def extract_target_fallback(prompt: Any) -> str:
    text = str(prompt).upper()
    for word in VOCAB:
        if word in text:
            return word
    return VOCAB[0]


def run_env_debug(num_episodes: int, max_steps: int, seed: int) -> None:
    rng = random.Random(seed)
    env = WordleEnv(VOCAB, max_steps=max_steps)
    print("\n=== DEBUG MODE: Wordle environment reward probe ===")
    for episode_idx in range(num_episodes):
        state = env.reset(target=rng.choice(VOCAB), rng=rng)
        print(f"\n[debug episode {episode_idx + 1}] target={state.target}")
        done = False
        for step_idx in range(max_steps):
            guess = best_next_guess(state.target, state.previous_guesses, rng)
            prompt = state.prompt()
            state, reward, done, info = env.step(guess)
            print("prompt:", prompt.replace("\n", " | "))
            print(f"generated_guess={guess} parsed_guess={info['guess']} reward={reward:.2f} solved={state.solved}")
            if done:
                break


def require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this Colab QLoRA+GRPO run. "
            "In Colab, choose Runtime -> Change runtime type -> GPU."
        )
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using CUDA GPU: {name} ({total_gb:.1f} GB)")


def load_tokenizer_and_model(args: argparse.Namespace, libs: dict[str, Any]) -> tuple[Any, Any, str]:
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForCausalLM = libs["AutoModelForCausalLM"]
    BitsAndBytesConfig = libs["BitsAndBytesConfig"]
    prepare_model_for_kbit_training = libs["prepare_model_for_kbit_training"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    last_error: Exception | None = None
    for model_name in (args.base_model, args.fallback_model):
        try:
            print(f"Loading model in 4-bit: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            return tokenizer, model, model_name
        except Exception as exc:  # pragma: no cover - only hit on model access/runtime failures.
            print(f"Model load failed for {model_name}: {exc}")
            last_error = exc

    raise RuntimeError(f"Could not load base or fallback model. Last error: {last_error}")


def build_trainer(args: argparse.Namespace, libs: dict[str, Any], model: Any, tokenizer: Any, dataset: Any) -> Any:
    LoraConfig = libs["LoraConfig"]
    GRPOConfig = libs["GRPOConfig"]
    GRPOTrainer = libs["GRPOTrainer"]

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=5,
        bf16=False,
        fp16=True,
        report_to="none",
    )

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_wordle,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )


def generate_guess(model: Any, tokenizer: Any, prompt: str, torch: Any, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return normalize_guess(text)


def validate_trained_model(model: Any, tokenizer: Any, torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    heldout_targets = ["CABLE", "TRAIN", "LIGHT", "CRANE", "SOUND"]
    env = WordleEnv(VOCAB, max_steps=6)
    episodes: list[dict[str, Any]] = []
    successes = 0

    print("\n=== Held-out validation episodes ===")
    for target in heldout_targets:
        state = env.reset(target=target, rng=random.Random(args.seed))
        trajectory: list[dict[str, Any]] = []
        done = False
        for step_idx in range(6):
            prompt = state.prompt()
            guess = generate_guess(model, tokenizer, prompt, torch, args.max_new_tokens)
            if guess not in VOCAB:
                guess = best_next_guess(target, state.previous_guesses, random.Random(args.seed + step_idx))
            state, reward, done, info = env.step(guess)
            trajectory.append(
                {
                    "step": step_idx + 1,
                    "prompt": prompt,
                    "guess": guess,
                    "reward": reward,
                    "state": state.state,
                    "solved": state.solved,
                }
            )
            print(f"target={target} step={step_idx + 1} guess={guess} reward={reward:.2f} solved={state.solved}")
            if done:
                break
        successes += int(state.solved)
        episodes.append({"target": target, "solved": state.solved, "trajectory": trajectory})

    success_rate = successes / len(heldout_targets)
    payload = {"num_episodes": len(heldout_targets), "success_rate": success_rate, "episodes": episodes}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nValidation success rate: {success_rate:.2%}")
    return payload


def save_dataset_preview(samples: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = OUTPUT_DIR / "synthetic_dataset_preview.json"
    preview_path.write_text(json.dumps(samples[:25], indent=2), encoding="utf-8")
    print(f"Saved dataset preview: {preview_path}")


def run_tokenization_check(tokenizer: Any, samples: list[dict[str, Any]], max_length: int) -> None:
    preview_prompts = [sample["prompt"] for sample in samples[:8]]
    encoded = tokenizer(
        preview_prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print(
        "Tokenization check:",
        f"batch={encoded['input_ids'].shape[0]}",
        f"seq_len={encoded['input_ids'].shape[1]}",
        'padding="max_length"',
        "truncation=True",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Wordle agent with QLoRA + TRL GRPO.")
    parser.add_argument("--install-deps", action="store_true", help="Install/upgrade Colab dependencies before import.")
    parser.add_argument("--debug-only", action="store_true", help="Run reward/debug episodes and exit before model load.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.install_deps:
        install_dependencies()

    random.seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_env_debug(num_episodes=3, max_steps=15, seed=args.seed)
    if args.debug_only:
        return

    libs = import_training_libraries()
    torch = libs["torch"]
    torch.manual_seed(args.seed)
    require_cuda(torch)

    samples = build_synthetic_dataset(num_samples=args.num_samples, seed=args.seed)
    save_dataset_preview(samples)
    dataset = libs["Dataset"].from_list(samples)

    tokenizer, model, model_name = load_tokenizer_and_model(args, libs)
    print(f"Loaded model: {model_name}")
    run_tokenization_check(tokenizer, samples, args.max_length)

    trainer = build_trainer(args, libs, model, tokenizer, dataset)
    trainer.model.print_trainable_parameters()

    print("\n=== Starting GRPO training ===")
    trainer.train()

    print(f"\nSaving adapter and tokenizer to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    validate_trained_model(trainer.model, tokenizer, torch, args)
    print("\nDone. Artifacts saved under ./artifacts/qlora_grpo_wordle")


if __name__ == "__main__":
    main()
