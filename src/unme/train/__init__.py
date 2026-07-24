"""Distillation training.

`losses.py` (sparse top-k KL + hidden-state match + combined term) is done and
gated by tests/test_losses.py. GLM Task 1 still owes `distill.py` — the training
loop that consumes `unme.data.dataset` batches and calls `combined_distill_loss`.
"""

from unme.train.losses import (
    combined_distill_loss,
    hidden_state_match_loss,
    topk_kl_loss,
)

__all__ = ["combined_distill_loss", "hidden_state_match_loss", "topk_kl_loss"]
