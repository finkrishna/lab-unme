"""ARCHITECT-OWNED acceptance test for the distillation losses.

The fleet implements `src/unme/train/losses.py` to make THIS file pass. Do not edit
this file to make it green — that defeats the gate. If a test looks wrong, flag it.

These pin the one place a cheap model most easily gets the math subtly wrong:
sparse (top-k) KL distillation. Requires torch (the `train` extra); skips cleanly
without it so the rest of the suite still runs.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from unme.train.losses import (
    hidden_state_match_loss,
    topk_kl_loss,
)


def _teacher_topk(logprobs_rows):
    """Build (ids, logprobs) tensors from python rows of log-probabilities."""
    lp = torch.tensor(logprobs_rows, dtype=torch.float32)
    k = lp.shape[-1]
    ids = torch.arange(k).expand(lp.shape)
    return ids, lp


def test_kl_is_zero_when_student_matches_teacher():
    # Teacher top-2 distribution over ids {0,1}: prob [0.75, 0.25].
    p = [0.75, 0.25]
    teacher_lp = [[math.log(p[0]), math.log(p[1])]]
    ids, tlp = _teacher_topk(teacher_lp)

    # Student full logits over vocab of 5; put teacher's mass on the same top-2 ids
    # and drive all other logits very negative so the renormalized top-k matches.
    V = 5
    student = torch.full((1, V), -30.0)
    student[0, 0] = math.log(p[0])
    student[0, 1] = math.log(p[1])

    loss = topk_kl_loss(student, ids, tlp, temperature=1.0)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_kl_positive_when_student_disagrees():
    teacher_lp = [[math.log(0.9), math.log(0.1)]]
    ids, tlp = _teacher_topk(teacher_lp)
    V = 5
    student = torch.full((1, V), -30.0)
    # Student says the opposite ordering.
    student[0, 0] = math.log(0.1)
    student[0, 1] = math.log(0.9)
    loss = topk_kl_loss(student, ids, tlp, temperature=1.0)
    assert loss.item() > 0.1


def test_kl_backprops():
    teacher_lp = [[math.log(0.6), math.log(0.4)]]
    ids, tlp = _teacher_topk(teacher_lp)
    student = torch.zeros((1, 6), requires_grad=True)
    loss = topk_kl_loss(student, ids, tlp, temperature=2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_kl_handles_batch_and_positions():
    # Shape (batch=2, positions=3, vocab=8), teacher top-4.
    B, T, V, K = 2, 3, 8, 4
    student = torch.randn(B, T, V, requires_grad=True)
    ids = torch.randint(0, V, (B, T, K))
    tlp = torch.log_softmax(torch.randn(B, T, K), dim=-1)
    loss = topk_kl_loss(student, ids, tlp, temperature=1.5)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_hidden_match_zero_when_equal():
    # Identity projection, identical hidden states -> zero loss.
    proj = torch.nn.Identity()
    h = torch.randn(2, 3, 16)
    loss = hidden_state_match_loss(h, h, proj)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hidden_match_projects_differing_dims():
    # Student dim 16 -> teacher dim 32 via a Linear projection; must run and be finite.
    proj = torch.nn.Linear(16, 32)
    hs = torch.randn(2, 3, 16)
    ht = torch.randn(2, 3, 32)
    loss = hidden_state_match_loss(hs, ht, proj)
    assert torch.isfinite(loss) and loss.item() > 0
