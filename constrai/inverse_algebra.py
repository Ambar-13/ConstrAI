"""
Inverse Effect Algebra — T7 Realization
=========================================

Automatically computes and applies inverse effects to enable exact rollback.

Theorem T7: undo(execute(s, a)) == s.

Proof strategy used here:
  State is immutable (formal.py §1). The snapshot state_before is stored
  alongside the executed action. Rollback applies inverse effects computed
  by diffing state_before and state_after. Because State is immutable,
  state_before is guaranteed to be unmodified and can be returned directly.

Use this module for:
  - Single-step rollback after a bad action
  - Recovery from capture basins (HJB barrier)
  - Compositional reasoning about reversibility of action sequences
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .formal import State, ActionSpec, Effect, _SENTINEL_DELETE


@dataclass(frozen=True)
class InverseEffect:
    """An effect paired with the prior state value it restores."""
    variable: str
    mode: str
    prior_value: object  # The value before the original effect
    
    def to_effect(self) -> Effect:
        """Convert to a standard Effect that restores the prior value."""
        if self.prior_value is _SENTINEL_DELETE:
            return Effect(self.variable, "delete")
        else:
            return Effect(self.variable, "set", self.prior_value)


@dataclass
class RollbackRecord:
    """
    Complete record needed to undo an action transition.
    
    Stored in the trace or checkpoint system for efficient recovery.
    """
    action_id: str
    action_name: str
    state_before_fingerprint: str
    state_before_snapshot: State
    state_after_fingerprint: str
    inverse_effects: Tuple[Effect, ...]
    timestamp: float
    
    def apply_rollback(self, current_state: State) -> State:
        """
        Apply rollback. Returns state_before (exact, via immutability).
        
        Theorem T7: Given that current_state == state_after, this returns
        a state equal to state_before (by object equality).
        """
        if current_state.fingerprint != self.state_after_fingerprint:
            raise ValueError(
                f"Cannot rollback: current state {current_state.fingerprint} "
                f"!= recorded after-state {self.state_after_fingerprint}"
            )
        # Apply inverse effects
        inv_action = ActionSpec(
            id=f"rollback_of_{self.action_id}",
            name=f"Rollback({self.action_name})",
            description="Automatic rollback to prior state",
            effects=self.inverse_effects,
            cost=0.0,  # Rollback is free
            reversible=True,
        )
        restored = inv_action.simulate(current_state)
        
        # Verify exactness (T7)
        if restored != self.state_before_snapshot:
            raise AssertionError(
                f"Rollback exactness (T7) failed! "
                f"Restored {restored.fingerprint} != "
                f"Original {self.state_before_snapshot.fingerprint}"
            )
        
        return restored


class InverseAlgebra:
    """
    Utility class for computing and managing inverse effects.
    
    The key insight: we store the state_before when an action is executed,
    so we can ALWAYS compute the exact inverse by diffing.
    """
    
    @staticmethod
    def compute_inverse_from_states(
        state_before: State,
        state_after: State,
        action: ActionSpec
    ) -> Tuple[Effect, ...]:
        """
        Compute inverse effects by comparing before and after states.
        
        This is the most robust method: it doesn't rely on Effect.apply()
        being reversible; it just looks at what actually changed.
        
        Theorem: This method ALWAYS produces the correct inverse, regardless
        of effect mode, because it's based on observed state deltas.
        """
        inverse_effects: List[Effect] = []
        
        # Find all keys that changed or were deleted
        all_keys = set(state_before.keys()) | set(state_after.keys())
        
        for key in all_keys:
            before_has = state_before.has(key)
            after_has = state_after.has(key)
            
            if not before_has and after_has:
                # Key was added → inverse is delete
                inverse_effects.append(Effect(key, "delete"))
            elif before_has and not after_has:
                # Key was deleted → inverse is restore
                inverse_effects.append(Effect(key, "set", state_before.get(key)))
            elif before_has and after_has:
                before_val = state_before.get(key)
                after_val = state_after.get(key)
                if before_val != after_val:
                    # Key changed → inverse restores old value
                    inverse_effects.append(Effect(key, "set", before_val))
        
        return tuple(inverse_effects)
    
    @staticmethod
    def make_rollback_record(
        action: ActionSpec,
        state_before: State,
        state_after: State,
        timestamp: float
    ) -> RollbackRecord:
        """Create a complete rollback record (snapshots + inverses)."""
        inverse_effects = InverseAlgebra.compute_inverse_from_states(
            state_before, state_after, action
        )
        return RollbackRecord(
            action_id=action.id,
            action_name=action.name,
            state_before_fingerprint=state_before.fingerprint,
            state_before_snapshot=state_before,
            state_after_fingerprint=state_after.fingerprint,
            inverse_effects=inverse_effects,
            timestamp=timestamp,
        )
    
    @staticmethod
    def verify_inverse_correctness(
        state_before: State,
        action: ActionSpec,
        inverse_effects: Tuple[Effect, ...]
    ) -> Tuple[bool, str]:
        """
        Test that the inverse effects actually restore the prior state.
        
        Returns (is_correct, diagnostic_message)
        """
        state_after = action.simulate(state_before)
        
        # Build and apply inverse action
        inverse_action = ActionSpec(
            id=f"verify_inverse_of_{action.id}",
            name=f"VerifyInverse({action.name})",
            description="Verification action",
            effects=inverse_effects,
            cost=0.0,
            reversible=True,
        )
        
        restored = inverse_action.simulate(state_after)
        
        if restored == state_before:
            return True, "Inverse correctness verified (T7 holds)"
        else:
            diffs = restored.diff(state_before)
            return False, f"Inverse failed: {len(diffs)} variables still differ: {diffs}"


# ═════════════════════════════════════════════════════════════════════════════
# Integration: Enhanced ActionSpec with built-in inverse support
# ═════════════════════════════════════════════════════════════════════════════

def action_with_inverse_guarantee(
    spec: ActionSpec,
    state_before: State
) -> Tuple[ActionSpec, Tuple[Effect, ...]]:
    """
    Wrapper that returns the action + precomputed inverse effects.
    
    Use this when you want to guarantee T7 from the start.
    """
    inverse_effects = InverseAlgebra.compute_inverse_from_states(
        state_before, spec.simulate(state_before), spec
    )
    
    # Verify
    is_correct, msg = InverseAlgebra.verify_inverse_correctness(
        state_before, spec, inverse_effects
    )
    
    if not is_correct:
        raise RuntimeError(f"Cannot guarantee inverse: {msg}")
    
    return spec, inverse_effects
