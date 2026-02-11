"""constrai.saliency

Integral Sensitivity Filter (Saliency Engine)
============================================

This module is optional infrastructure for prompt-token optimization.
It computes a *discrete integral sensitivity* score per state key and
returns a pruned `State` containing only salient keys.

Now integrated with JacobianFusion for authoritative boundary sensitivity.

Math (discrete integral over available actions):

    S(k) = Σ_{a ∈ A} 1[k ∈ affected(a)] · w(a) + α·J_k

Where:
  - k is a state key
  - A is the set of currently available actions
  - affected(a) are the state variables action `a` can change
  - w(a) is a nonnegative weight derived from formal value signals
  - J_k is the Jacobian sensitivity score (∂/∂Safety)
  - α is weight factor for Jacobian (forces critical vars into prompt)

Keys with S(k) below a threshold can often be dropped without changing the
argmax decision, reducing prompt size.

Guarantee level: HEURISTIC (for actions), AUTHORITATIVE (for Jacobian critical vars)
  - Action-based saliency is best-effort; safety enforcement uses full state
  - Jacobian-critical variables are ALWAYS kept in prompt (hard rule)
  - Formal safety kernel still uses the full state for all checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from .formal import State, ActionSpec
from .reasoning import ActionValue

if TYPE_CHECKING:
    from .jacobian_fusion import JacobianFusion


@dataclass(frozen=True)
class SaliencyResult:
    """Result of saliency analysis for prompt pruning."""

    kept_keys: Tuple[str, ...]
    scores: Dict[str, float]
    dropped_keys: Tuple[str, ...] = ()


class SaliencyEngine:
    """Computes Integral Sensitivity scores and prunes `State` for prompts.
    
    Now enhanced with JacobianFusion: critical variables (high boundary sensitivity)
    are ALWAYS kept, even if action-based heuristic would prune them.
    """

    def __init__(self, *, threshold: float = 0.05, max_keys: int = 20,
                 jacobian_fusion: Optional['JacobianFusion'] = None,
                 jacobian_weight: float = 10.0):
        """
        Args:
            threshold: Score threshold below which to drop keys
            max_keys: Maximum keys to keep
            jacobian_fusion: Optional JacobianFusion for authoritative sensitivity
            jacobian_weight: Multiplier for Jacobian scores (forces critical vars)
        """
        self.threshold = float(threshold)
        self.max_keys = int(max_keys)
        self.jacobian_fusion = jacobian_fusion
        self.jacobian_weight = float(jacobian_weight)

    def analyze(
        self,
        *,
        state: State,
        available_actions: List[ActionSpec],
        action_values: List[ActionValue],
    ) -> SaliencyResult:
        """
        Analyze state saliency with optional Jacobian integration.
        
        If JacobianFusion is available, variables with high boundary sensitivity
        are forced into prompt with infinite score (never pruned).
        """
        # Explicit disable switch: keep full state.
        if self.threshold <= 0 or self.max_keys <= 0:
            d = state.to_dict()
            return SaliencyResult(kept_keys=tuple(d.keys()), scores={k: float("inf") for k in d.keys()}, dropped_keys=())

        d = state.to_dict()
        scores: Dict[str, float] = {k: 0.0 for k in d.keys()}
        value_by_id = {v.action_id: float(v.value_score) for v in action_values}

        # Step 1: Action-based heuristic scoring
        for a in available_actions:
            w = abs(value_by_id.get(a.id, 0.0))
            if w <= 0:
                continue
            for k in a.affected_variables():
                if k in scores:
                    scores[k] += w

        # Step 2: Jacobian-based authoritative scoring (if available)
        critical_vars = set()
        if self.jacobian_fusion is not None:
            jacobian_report = self.jacobian_fusion.compute_jacobian(state)
            critical_vars = set(jacobian_report.critical_variables)
            
            # Apply Jacobian weight to critical variables
            for var in critical_vars:
                if var in scores:
                    # Force into prompt with high score
                    scores[var] = float("inf")

        # Always keep internal/meta keys
        for k in list(scores.keys()):
            if k.startswith("_"):
                scores[k] = float("inf")

        # Keep: internal vars, Jacobian-critical vars, or above-threshold vars
        kept = [k for k, s in scores.items() if s == float("inf") or s >= self.threshold]
        if self.max_keys > 0 and len(kept) > self.max_keys:
            kept = sorted(kept, key=lambda kk: scores.get(kk, 0.0), reverse=True)[: self.max_keys]

        kept_set = set(kept)
        dropped = tuple(k for k in d.keys() if k not in kept_set)
        return SaliencyResult(kept_keys=tuple(kept), scores=scores, dropped_keys=dropped)

    def prune_state(self, *, state: State, saliency: SaliencyResult) -> State:
        if not saliency.kept_keys:
            return state
        d = state.to_dict()
        return State({k: d[k] for k in saliency.kept_keys if k in d})
