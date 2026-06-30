"""
═══════════════════════════════════════════════════════════════════════════════
  Deep Knowledge Tracing — LSTM Model
  Tracks student mastery per concept over time using sequential interactions.
  Architecture: Embedding → LSTM → Dropout → Linear → Sigmoid
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

logger = logging.getLogger(__name__)


class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing with LSTM.

    Input:  Sequence of (concept_id, is_correct) pairs per student
    Output: Mastery probability per concept at each timestep

    Architecture:
        1. Separate embeddings for concept and response
        2. Concatenated → LSTM layers with dropout
        3. Linear projection → Sigmoid → per-concept mastery probability
    """

    def __init__(
        self,
        num_concepts: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.num_concepts = num_concepts
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layers for concept and response
        # concept_id → embedding, response (0 or 1) → embedding
        self.concept_embedding = nn.Embedding(
            num_embeddings=num_concepts,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.response_embedding = nn.Embedding(
            num_embeddings=2,  # correct / incorrect
            embedding_dim=embedding_dim,
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,   # concept + response concatenated
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_concepts)
        self.sigmoid = nn.Sigmoid()

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"DKT Model initialized: {num_concepts} concepts, "
            f"{hidden_size} hidden, {num_layers} layers, "
            f"emb_dim={embedding_dim}"
        )

    def _init_weights(self):
        """Xavier/Kaiming initialization for better convergence."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.1)

        # Initialize output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            concept_ids: (batch, seq_len) — concept indices
            responses:   (batch, seq_len) — binary correctness (0/1)
            lengths:     (batch,) — actual sequence lengths (for packing)

        Returns:
            output: (batch, seq_len, num_concepts) — mastery probabilities
        """
        batch_size, seq_len = concept_ids.shape

        # Embed concepts and responses
        concept_emb = self.concept_embedding(concept_ids)    # (B, T, E)
        response_emb = self.response_embedding(responses)    # (B, T, E)

        # Concatenate: (B, T, 2E)
        x = torch.cat([concept_emb, response_emb], dim=-1)

        # Pack sequences for efficient LSTM processing
        if lengths is not None:
            # Ensure lengths are on CPU and sorted for packing
            lengths_cpu = lengths.cpu().clamp(min=1)
            x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Unpack
        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)

        # Layer norm + dropout
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Project to concept space
        output = self.fc(lstm_out)    # (B, T, num_concepts)
        output = self.sigmoid(output)

        return output

    def predict_mastery(
        self,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Predict current mastery state for a student.
        Returns the mastery vector at the last valid timestep.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(concept_ids, responses, lengths)

            if lengths is not None:
                # Get the output at the last valid timestep for each student
                batch_size = output.shape[0]
                mastery = torch.zeros(batch_size, self.num_concepts)
                for i in range(batch_size):
                    last_idx = int(lengths[i].item()) - 1
                    mastery[i] = output[i, last_idx]
            else:
                mastery = output[:, -1]  # Last timestep

        return mastery.numpy()

    def get_knowledge_state(
        self,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Get detailed knowledge state including mastery per concept and hidden state.
        Used by the agent for comprehensive analysis.
        """
        self.eval()
        with torch.no_grad():
            # Get embeddings
            concept_emb = self.concept_embedding(concept_ids)
            response_emb = self.response_embedding(responses)
            x = torch.cat([concept_emb, response_emb], dim=-1)

            # LSTM forward — capture hidden states
            lstm_out, (h_n, c_n) = self.lstm(x)
            lstm_out = self.layer_norm(lstm_out)

            # Final mastery output
            output = self.sigmoid(self.fc(lstm_out))

            if lengths is not None:
                batch_idx = 0
                last_idx = int(lengths[batch_idx].item()) - 1
                mastery = output[batch_idx, last_idx].numpy()
                trajectory = output[batch_idx, :last_idx + 1].numpy()
            else:
                mastery = output[0, -1].numpy()
                trajectory = output[0].numpy()

        return {
            "mastery": mastery,                    # (num_concepts,) — current mastery
            "trajectory": trajectory,              # (seq_len, num_concepts) — mastery over time
            "hidden_state": h_n[-1, 0].numpy(),    # (hidden_size,) — latent knowledge state
        }


class DKTDataset(torch.utils.data.Dataset):
    """
    Dataset for DKT training.
    Converts raw interaction sequences into padded tensors.
    """

    def __init__(self, sequences: list[list[tuple]], num_concepts: int):
        """
        Args:
            sequences: List of [(concept_id, is_correct), ...] per student
            num_concepts: Total number of unique concepts
        """
        self.sequences = sequences
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        concept_ids = [s[0] for s in seq]
        responses = [s[1] for s in seq]
        length = len(seq)

        return {
            "concept_ids": torch.tensor(concept_ids, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "length": length,
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate with padding for variable-length sequences."""
        concept_ids = [item["concept_ids"] for item in batch]
        responses = [item["responses"] for item in batch]
        lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

        # Pad sequences
        concept_ids_padded = pad_sequence(concept_ids, batch_first=True, padding_value=0)
        responses_padded = pad_sequence(responses, batch_first=True, padding_value=0)

        return {
            "concept_ids": concept_ids_padded,
            "responses": responses_padded,
            "lengths": lengths,
        }
