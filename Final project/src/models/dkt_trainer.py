"""
═══════════════════════════════════════════════════════════════════════════════
  DKT Trainer — Training loop, evaluation, checkpointing, early stopping
  Handles the full lifecycle of training the Deep Knowledge Tracing model.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DKT_CONFIG, MODELS_DIR, TRAIN_RATIO, VAL_RATIO
from src.models.dkt_model import DKTModel, DKTDataset

logger = logging.getLogger(__name__)


class DKTTrainer:
    """
    Full training pipeline for the DKT model.

    Features:
        - Train/val/test split with early stopping
        - Gradient clipping, learning rate scheduling
        - Model checkpointing (best + periodic)
        - Comprehensive metrics: AUC-ROC, accuracy, RMSE
        - Training history for visualization
    """

    def __init__(
        self,
        num_concepts: int,
        device: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.config = config or DKT_CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_concepts = num_concepts

        # Initialize model
        self.model = DKTModel(
            num_concepts=num_concepts,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            embedding_dim=self.config["embedding_dim"],
        ).to(self.device)

        # Loss function
        self.criterion = nn.BCELoss(reduction="none")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

        # Training history
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_auc": [], "val_auc": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }

        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"DKT Trainer initialized on {self.device}")
        logger.info(f"  Parameters: {total_params:,}")

    def train(
        self,
        sequences: list[list[tuple]],
        save_dir: Optional[Path] = None,
    ) -> dict:
        """
        Full training pipeline.

        Args:
            sequences: List of interaction sequences [(concept_id, is_correct), ...]
            save_dir: Directory to save model checkpoints

        Returns:
            dict with training history and best metrics
        """
        save_dir = save_dir or MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset
        dataset = DKTDataset(sequences, self.num_concepts)

        # Split: train / val / test
        n_total = len(dataset)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Data split: train={n_train}, val={n_val}, test={n_test}")

        # Data loaders
        train_loader = DataLoader(
            train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=DKTDataset.collate_fn,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=DKTDataset.collate_fn,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=DKTDataset.collate_fn,
            num_workers=0,
        )

        # Training loop
        logger.info("━" * 60)
        logger.info("Starting DKT Training")
        logger.info("━" * 60)

        start_time = time.time()

        for epoch in range(1, self.config["epochs"] + 1):
            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._evaluate(val_loader)

            # Update scheduler
            self.scheduler.step(val_metrics["auc"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_auc"].append(train_metrics["auc"])
            self.history["val_auc"].append(val_metrics["auc"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(current_lr)

            # Logging
            logger.info(
                f"  Epoch {epoch:3d}/{self.config['epochs']} | "
                f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f}/{val_metrics['auc']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Best model checkpoint
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(save_dir / "dkt_best.pt", epoch, val_metrics)
                logger.info(f"  ★ New best AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"  ⏹ Early stopping at epoch {epoch} (patience={self.config['early_stopping_patience']})")
                break

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s")

        # Load best model and evaluate on test set
        self._load_checkpoint(save_dir / "dkt_best.pt")
        test_metrics = self._evaluate(test_loader)

        logger.info("━" * 60)
        logger.info("Final Test Metrics:")
        logger.info(f"  AUC-ROC:  {test_metrics['auc']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  RMSE:     {test_metrics['rmse']:.4f}")
        logger.info(f"  Best epoch: {self.best_epoch}")
        logger.info("━" * 60)

        return {
            "history": self.history,
            "test_metrics": test_metrics,
            "best_epoch": self.best_epoch,
            "training_time": elapsed,
        }

    def _train_epoch(self, loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        n_batches = 0

        for batch in loader:
            concept_ids = batch["concept_ids"].to(self.device)
            responses = batch["responses"].to(self.device)
            lengths = batch["lengths"]

            # Forward pass
            output = self.model(concept_ids, responses, lengths)

            # Create target: next-step prediction
            # For each timestep t, predict the response at t+1
            target_concepts = concept_ids[:, 1:]      # Concepts at t+1
            target_responses = responses[:, 1:].float()  # Responses at t+1
            pred_output = output[:, :-1]               # Predictions from t

            # Gather predictions for the specific concepts being tested
            batch_size, seq_len, _ = pred_output.shape
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
            pred_for_concept = pred_output[batch_indices, torch.arange(seq_len).unsqueeze(0), target_concepts]

            # Create mask for valid positions (not padded)
            mask = torch.zeros_like(target_responses)
            for i, length in enumerate(lengths):
                valid_len = min(int(length.item()) - 1, seq_len)
                if valid_len > 0:
                    mask[i, :valid_len] = 1.0

            # Masked loss
            loss = self.criterion(pred_for_concept, target_responses)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions for metrics
            valid_mask = mask.bool().cpu()
            preds = pred_for_concept.detach().cpu()[valid_mask].numpy()
            targets = target_responses.cpu()[valid_mask].numpy()
            if len(preds) > 0:
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())

        return self._compute_metrics(all_preds, all_targets, total_loss / max(n_batches, 1))

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        n_batches = 0

        for batch in loader:
            concept_ids = batch["concept_ids"].to(self.device)
            responses = batch["responses"].to(self.device)
            lengths = batch["lengths"]

            output = self.model(concept_ids, responses, lengths)

            target_concepts = concept_ids[:, 1:]
            target_responses = responses[:, 1:].float()
            pred_output = output[:, :-1]

            batch_size, seq_len, _ = pred_output.shape
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
            pred_for_concept = pred_output[batch_indices, torch.arange(seq_len).unsqueeze(0), target_concepts]

            mask = torch.zeros_like(target_responses)
            for i, length in enumerate(lengths):
                valid_len = min(int(length.item()) - 1, seq_len)
                if valid_len > 0:
                    mask[i, :valid_len] = 1.0

            loss = self.criterion(pred_for_concept, target_responses)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            total_loss += loss.item()
            n_batches += 1

            valid_mask = mask.bool().cpu()
            preds = pred_for_concept.cpu()[valid_mask].numpy()
            targets = target_responses.cpu()[valid_mask].numpy()
            if len(preds) > 0:
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())

        return self._compute_metrics(all_preds, all_targets, total_loss / max(n_batches, 1))

    @staticmethod
    def _compute_metrics(preds: list, targets: list, loss: float) -> dict:
        """Compute AUC-ROC, accuracy, and RMSE."""
        if len(preds) == 0 or len(set(targets)) < 2:
            return {"loss": loss, "auc": 0.5, "accuracy": 0.0, "rmse": 1.0}

        preds_arr = np.array(preds)
        targets_arr = np.array(targets)

        try:
            auc = roc_auc_score(targets_arr, preds_arr)
        except ValueError:
            auc = 0.5

        binary_preds = (preds_arr >= 0.5).astype(int)
        acc = accuracy_score(targets_arr.astype(int), binary_preds)
        rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))

        return {"loss": loss, "auc": auc, "accuracy": acc, "rmse": rmse}

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "num_concepts": self.num_concepts,
        }, path)

    def _load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    @classmethod
    def load_trained_model(
        cls,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> "DKTTrainer":
        """Load a trained model from disk."""
        path = checkpoint_path or MODELS_DIR / "dkt_best.pt"
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        trainer = cls(
            num_concepts=checkpoint["num_concepts"],
            device=device,
            config=checkpoint["config"],
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()

        logger.info(
            f"Loaded trained DKT model: {checkpoint['num_concepts']} concepts, "
            f"epoch {checkpoint['epoch']}, AUC={checkpoint['metrics']['auc']:.4f}"
        )
        return trainer
