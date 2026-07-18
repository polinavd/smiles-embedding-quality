"""Fine-tune BERT/RoBERTa with the unsupervised SimCSE objective.

This is a modern PyTorch/Hugging Face implementation of the core method from:
    Gao, Yao, and Chen (2021), "SimCSE: Simple Contrastive Learning of
    Sentence Embeddings".

For every sentence, the same tokenized input is encoded twice in one forward
pass. Standard Transformer dropout creates the two positive views. Other
sentences in the physical mini-batch are negatives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


@dataclass(frozen=True)
class TrainSettings:
    experiment_name: str
    seed: int
    model_name: str
    pooler_type: str
    mlp_only_train: bool
    max_seq_length: int
    dropout: float
    train_file: Path
    max_samples: int | None
    epochs: int
    batch_size: int
    learning_rate: float
    temperature: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    dataloader_num_workers: int
    logging_steps: int
    requested_device: str
    output_dir: Path
    results_file: Path
    overwrite_output_dir: bool


class TextLineDataset(Dataset[str]):
    """One non-empty sentence per line, matching the SimCSE Wikipedia file."""

    def __init__(self, path: Path, max_samples: int | None = None) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Training file not found: {path}")

        sentences: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sentence = line.strip()
                if not sentence:
                    continue
                sentences.append(sentence)
                if max_samples is not None and len(sentences) >= max_samples:
                    break

        if len(sentences) < 2:
            raise ValueError("The training file must contain at least two sentences.")
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> str:
        return self.sentences[index]


class SimCSEModel(nn.Module):
    """Pretrained encoder plus the paper's train-time CLS MLP projection."""

    def __init__(self, encoder: nn.Module, hidden_size: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.encoder = encoder
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.temperature = temperature
        self._initialize_projection(hidden_size)

    def _initialize_projection(self, hidden_size: int) -> None:
        linear = self.projection[0]
        initializer_range = float(getattr(self.encoder.config, "initializer_range", 0.02))
        nn.init.normal_(linear.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(linear.bias)
        if linear.in_features != hidden_size or linear.out_features != hidden_size:
            raise RuntimeError("Projection size does not match encoder hidden size.")

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Two identical token views in one forward pass. Dropout masks are sampled
        # independently for each duplicated example.
        doubled = {key: torch.cat((value, value), dim=0) for key, value in batch.items()}
        outputs = self.encoder(**doubled, return_dict=True)

        # Paper baseline: raw [CLS] hidden state, followed by a train-time MLP.
        cls_embeddings = outputs.last_hidden_state[:, 0]
        projected = self.projection(cls_embeddings)
        view_1, view_2 = projected.chunk(2, dim=0)

        view_1 = F.normalize(view_1, p=2, dim=-1)
        view_2 = F.normalize(view_2, p=2, dim=-1)

        # Each row i must identify its second dropout view at column i.
        logits = torch.matmul(view_1, view_2.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping.")
    return data


def parse_settings(config: dict[str, Any]) -> TrainSettings:
    try:
        experiment = config["experiment"]
        model = config["model"]
        data = config["data"]
        training = config["training"]
        output = config["output"]
    except KeyError as exc:
        raise KeyError(f"Missing top-level configuration section: {exc.args[0]}") from exc

    settings = TrainSettings(
        experiment_name=str(experiment["name"]),
        seed=int(experiment["seed"]),
        model_name=str(model["name_or_path"]),
        pooler_type=str(model.get("pooler_type", "cls")),
        mlp_only_train=bool(model.get("mlp_only_train", True)),
        max_seq_length=int(model.get("max_seq_length", 32)),
        dropout=float(model.get("dropout", 0.1)),
        train_file=Path(data["train_file"]),
        max_samples=(None if data.get("max_samples") is None else int(data["max_samples"])),
        epochs=int(training["num_train_epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        temperature=float(training["temperature"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        logging_steps=int(training.get("logging_steps", 20)),
        requested_device=str(training.get("device", "auto")),
        output_dir=Path(output["model_dir"]),
        results_file=Path(output["results_file"]),
        overwrite_output_dir=bool(output.get("overwrite_output_dir", False)),
    )

    if settings.pooler_type != "cls":
        raise ValueError("This paper-baseline trainer currently supports pooler_type='cls' only.")
    if not settings.mlp_only_train:
        raise ValueError("Unsupervised paper replication expects mlp_only_train: true.")
    if settings.batch_size < 2:
        raise ValueError("SimCSE needs batch_size >= 2 to provide in-batch negatives.")
    if settings.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    return settings


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Set overwrite_output_dir: true to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_collator(tokenizer: Any, max_seq_length: int):
    def collate(sentences: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

    return collate


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def save_cpu_state_dict(module: nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    torch.save(state, path)


def train(settings: TrainSettings) -> dict[str, Any]:
    set_reproducible_seed(settings.seed)
    device = choose_device(settings.requested_device)
    prepare_output_directory(settings.output_dir, settings.overwrite_output_dir)
    settings.results_file.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    dataset = TextLineDataset(settings.train_file, settings.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)

    hf_config = AutoConfig.from_pretrained(settings.model_name)
    # BERT and RoBERTa use these standard dropout fields.
    if hasattr(hf_config, "hidden_dropout_prob"):
        hf_config.hidden_dropout_prob = settings.dropout
    if hasattr(hf_config, "attention_probs_dropout_prob"):
        hf_config.attention_probs_dropout_prob = settings.dropout

    encoder = AutoModel.from_pretrained(settings.model_name, config=hf_config)
    model = SimCSEModel(
        encoder=encoder,
        hidden_size=int(hf_config.hidden_size),
        temperature=settings.temperature,
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=make_collator(tokenizer, settings.max_seq_length),
        num_workers=settings.dataloader_num_workers,
        generator=generator,
        pin_memory=(device.type == "cuda"),
    )
    if len(loader) == 0:
        raise RuntimeError("No full batch was produced. Lower batch_size or add more sentences.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / settings.gradient_accumulation_steps)
    total_updates = updates_per_epoch * settings.epochs
    warmup_steps = int(total_updates * settings.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    update_step = 0
    losses: list[float] = []
    started = time.perf_counter()

    print(f"Experiment: {settings.experiment_name}")
    print(f"Device: {device}")
    print(f"Sentences: {len(dataset):,}")
    print(f"Physical batch size: {settings.batch_size}")
    print(f"In-batch negatives per anchor: {settings.batch_size - 1}")

    for epoch in range(settings.epochs):
        epoch_loss = 0.0
        for batch_index, cpu_batch in enumerate(loader, start=1):
            global_step += 1
            batch = move_batch(cpu_batch, device)
            loss, _ = model(batch)
            raw_loss = float(loss.detach().cpu())
            losses.append(raw_loss)
            epoch_loss += raw_loss

            (loss / settings.gradient_accumulation_steps).backward()
            should_update = (
                batch_index % settings.gradient_accumulation_steps == 0 or batch_index == len(loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if settings.logging_steps > 0 and global_step % settings.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                recent = losses[-settings.logging_steps :]
                print(
                    f"epoch={epoch + 1}/{settings.epochs} "
                    f"batch={batch_index}/{len(loader)} "
                    f"loss={sum(recent) / len(recent):.6f} "
                    f"lr={current_lr:.3e}"
                )

        print(f"epoch={epoch + 1} mean_loss={epoch_loss / len(loader):.6f}")
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    elapsed = time.perf_counter() - started

    # For unsupervised SimCSE, the train-time MLP is removed at test time.
    model.encoder.save_pretrained(settings.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(settings.output_dir)
    save_cpu_state_dict(model.projection, settings.output_dir / "train_time_projection.pt")

    metadata = {
        "method": "unsupervised_simcse",
        "paper": "Gao et al. (2021)",
        "experiment_name": settings.experiment_name,
        "base_model": settings.model_name,
        "pooler_type": settings.pooler_type,
        "mlp_only_train": settings.mlp_only_train,
        "evaluation_pooling": "cls_before_pooler",
        "temperature": settings.temperature,
        "dropout": settings.dropout,
        "max_seq_length": settings.max_seq_length,
        "physical_batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "seed": settings.seed,
    }
    with (settings.output_dir / "simcse_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = {
        **metadata,
        "device": str(device),
        "num_sentences": len(dataset),
        "epochs": settings.epochs,
        "optimizer_updates": update_step,
        "mean_training_loss": float(np.mean(losses)),
        "last_training_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "output_dir": str(settings.output_dir),
    }
    with settings.results_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved fine-tuned encoder to: {settings.output_dir}")
    print(f"Saved training summary to: {settings.results_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="YAML experiment configuration.",
    )
    args = parser.parse_args()
    settings = parse_settings(load_yaml(args.config))
    train(settings)


if __name__ == "__main__":
    main()