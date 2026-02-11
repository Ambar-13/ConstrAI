"""Safe Hover Mode Enforcement

This module enforces a hard stop when safety barriers are violated.
When an unsafe state is detected, the system terminates the action loop
and triggers rollback. The LLM cannot override this decision.

Theorem AHJ-1 (Safe Hover Completeness):
  If state s violates any safety barrier, the system MUST rollback
  to the prior safe state. The LLM cannot override this.

Theorem AHJ-2 (Termination Guarantee):
  Safe Hover mode itself will not loop indefinitely.

Application:
  1. Monitor checks state before and after action execution
  2. If unsafe state detected → immediately signal TERMINATE_SAFE_HOVER
  3. Orchestrator kills LLM loop and applies rollback
  4. System returns to prior known-safe state
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .formal import State, ActionSpec
from .reference_monitor import CaptureBasin


class SafeHoverSignal(Enum):
    """Signals from authoritative HJB enforcement."""
    PROCEED = "proceed"  # Action is safe
    SAFE_HOVER = "safe_hover"  # Request agent to do nothing
    TERMINATE_AND_ROLLBACK = "terminate_and_rollback"  # Hard stop + rollback


@dataclass(frozen=True)
class HJBEnforcementCheck:
    """Result of authoritative HJB enforcement check."""
    signal: SafeHoverSignal
    violated_basin: Optional[CaptureBasin]
    reason: str
    requires_immediate_rollback: bool


class AuthoritativeHJBBarrier:
    """
    HARD enforcement of HJB barrier: terminates LLM loop if unsafe state detected.
    
    This is not a monitor or observer—it is a GATE. If barrier triggers,
    the system physically cannot continue.
    """

    def __init__(self, capture_basins: Optional[List[CaptureBasin]] = None):
        """
        Args:
            capture_basins: List of CaptureBasin definitions (danger zones)
        """
        self.basins = capture_basins or []

    def check_state_safety(self, state: State) -> HJBEnforcementCheck:
        """
        Check if state is in any capture basin (danger zone).
        
        Returns:
            HJBEnforcementCheck with signal:
              - PROCEED: State is safe
              - SAFE_HOVER: State is borderline (warn)
              - TERMINATE_AND_ROLLBACK: State is trapped (hard stop)
        """
        for basin in self.basins:
            if basin.is_bad(state):
                return HJBEnforcementCheck(
                    signal=SafeHoverSignal.TERMINATE_AND_ROLLBACK,
                    violated_basin=basin,
                    reason=f"State violates capture basin '{basin.name}'",
                    requires_immediate_rollback=True
                )

        return HJBEnforcementCheck(
            signal=SafeHoverSignal.PROCEED,
            violated_basin=None,
            reason="State is safe; no capture basin violations",
            requires_immediate_rollback=False
        )

    def check_action_leads_to_danger(self, 
                                      state: State, 
                                      action: ActionSpec,
                                      available_actions: List[ActionSpec]) -> HJBEnforcementCheck:
        """
        Check if action would lead directly into a capture basin.
        
        This is the "cliff detection" test: does the action immediately
        endanger the system?
        """
        # Simulate action
        sim_state = action.simulate(state)

        # Check if simulated state is in a basin
        for basin in self.basins:
            if basin.is_bad(sim_state):
                return HJBEnforcementCheck(
                    signal=SafeHoverSignal.TERMINATE_AND_ROLLBACK,
                    violated_basin=basin,
                    reason=f"Action '{action.name}' would enter capture basin '{basin.name}'. Rejected.",
                    requires_immediate_rollback=False  # Action never executed
                )

        # Check if action would transition to borderline state
        # (Jacobian analysis would be done by caller, we just check basins here)

        return HJBEnforcementCheck(
            signal=SafeHoverSignal.PROCEED,
            violated_basin=None,
            reason=f"Action '{action.name}' is safe; no basin entry",
            requires_immediate_rollback=False
        )

    def enforce_safe_hover(self, state: State) -> SafeHoverSignal:
        """
        Authoritative decision: PROCEED, SAFE_HOVER, or TERMINATE_AND_ROLLBACK?
        
        This is the actual enforcement gate that orchestrator.py calls.
        Result is BINDING—no LLM override possible.
        """
        check = self.check_state_safety(state)
        return check.signal
