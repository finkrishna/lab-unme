"""Distillation losses: sparse top-k KL, hidden-state match, and the combined term.

The teacher emits logits SPARSE (top-k per position, as log-softmax values — see
``unme.schemas.StepLogits``). We gather the student's full-vocab logits on those
ids, temperature-soften the student over the k-support only and renormalize, then
take ``KL(teacher_probs || student_topk)``. Hidden match projects the student's
hidden states up to the teacher dimension and minimizes MSE.

All losses honour an optional ``mask`` (1 = keep a position, 0 = pad). Means are
taken over valid positions only; batches that are fully masked return a zero scalar
that still carries a finite grad graph.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "unme.train.losses requires torch. Install the train extra: "
        "pip install 'lab-unme[train]'"
    ) from e


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of ``values`` over all dims, weighted by a (broadcastable) mask.

    ``mask`` is 1=keep, 0=ignore. A fully-masked batch returns 0.0 (with a grad graph
    so optimizers don't choke). ``mask`` is broadcast onto ``values``' shape via
    ``torch.broadcast_to`` (so a per-position mask ``[B, T]`` or ``[B, T, 1]`` applies
    straightforwardly to per-position scalars ``[B, T]`` or per-token tensors
    ``[B, T, K]``).
    """
    if mask is None:
        return values.mean()
    m = mask.to(values.dtype)
    if m.shape != values.shape:
        m = torch.broadcast_to(m, values.shape)
    summed = (values * m).sum()
    denom = m.sum().clamp_min(1.0)
    return summed / denom


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_top_ids: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    temperature: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(teacher top-k || student top-k), mean over valid positions, scaled by T^2.

    Args:
        student_logits: (..., V) full-vocab student logits.
        teacher_top_ids: (..., K) ids of the teacher's top-k tokens per position.
        teacher_top_logprobs: (..., K) teacher log-probs for those ids (already
            log-softmax over the full vocab; renormalized over the k-support here).
        temperature: distillation temperature T. The loss is multiplied by T*T.
        mask: optional (..., ) with 1 on real positions, 0 on pad. Mean is over
            kept positions; a fully-masked batch returns 0.0.

    Returns:
        Scalar tensor.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    t = float(temperature)

    # Gather student logits on the teacher top-k support.
    student_topk = torch.gather(student_logits, dim=-1, index=teacher_top_ids)
    # Renormalize the student over the k-support, softened by T.
    student_log_p = F.log_softmax(student_topk / t, dim=-1)
    # Teacher top-k probabilities renormalized over the k-support (conditional mass).
    teacher_p = torch.softmax(teacher_top_logprobs, dim=-1)
    teacher_log_p = torch.log(teacher_p.clamp_min(torch.finfo(teacher_p.dtype).tiny))

    # KL(p_t || p_s) = sum_k p_t * (log p_t - log p_s) over the k support, per position.
    kl_per_position = (teacher_p * (teacher_log_p - student_log_p)).sum(dim=-1)

    loss = _masked_mean(kl_per_position, mask) * (t * t)
    return loss


def hidden_state_match_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection: Any,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE(projection(student_hidden), teacher_hidden), mean over valid positions.

    Args:
        student_hidden: (..., D_s) student hidden states per position.
        teacher_hidden: (..., D_t) teacher hidden states per position.
        projection: callable / nn.Module mapping D_s -> D_t (e.g. nn.Linear).
        mask: optional (..., ) with 1 on real positions, 0 on pad.

    Returns:
        Scalar MSE over kept positions.
    """
    projected = projection(student_hidden)
    se = (projected - teacher_hidden).pow(2)
    # Mean over every trailing dim (incl. hidden) per position, then mask-mean.
    per_position = se.flatten(start_dim=-1).mean(dim=-1)
    return _masked_mean(per_position, mask)


def combined_distill_loss(
    student_logits: torch.Tensor,
    output_ids: torch.Tensor,
    teacher_top_ids: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    student_hidden: torch.Tensor | None = None,
    teacher_hidden: torch.Tensor | None = None,
    projection: Any = None,
    *,
    alpha_kl: float,
    alpha_hidden: float,
    alpha_ce: float,
    temperature: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total distillation objective = alpha_kl*KL + alpha_hidden*hidden + alpha_ce*CE.

    The hidden term is added only when both ``student_hidden`` and ``teacher_hidden``
    (plus ``projection``) are supplied. The hard-label cross-entropy is taken against
    ``output_ids`` on the student's full vocab, masked by ``mask``.

    Returns:
        (total, {"topk_kl": ..., "hidden": ... or 0., "ce": ...}).
    """
    terms: dict[str, torch.Tensor] = {}

    kl = topk_kl_loss(
        student_logits,
        teacher_top_ids,
        teacher_top_logprobs,
        temperature=temperature,
        mask=mask,
    )
    terms["topk_kl"] = kl

    # Cross-entropy on the hard labels (full vocab), masked.
    # student_logits: (..., V) ; output_ids: (...,) -> gather per-position logp.
    log_probs = F.log_softmax(student_logits, dim=-1)
    ids = output_ids.unsqueeze(-1).expand(*output_ids.shape, 1)
    token_log_p = torch.gather(log_probs, dim=-1, index=ids).squeeze(-1)
    ce = _masked_mean(-token_log_p, mask)
    terms["ce"] = ce

    # Hidden match only when all three pieces are present.
    hidden_term: torch.Tensor
    if student_hidden is not None and teacher_hidden is not None and projection is not None:
        hidden_term = hidden_state_match_loss(
            student_hidden, teacher_hidden, projection, mask=mask
        )
    else:
        hidden_term = torch.zeros((), dtype=student_logits.dtype, device=student_logits.device)
    terms["hidden"] = hidden_term

    total = alpha_kl * kl + alpha_hidden * hidden_term + alpha_ce * ce
    return total, terms
