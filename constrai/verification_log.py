"""Verification Log - Proof Records

This module records what was checked during execution and what passed.
The log can be verified offline by a third party.

Each run emits a JSON file containing:
- What invariants were checked
- What passed/failed
- Hash links to the execution trace
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .formal import SafetyVerdict


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class ProofStep:
    step_index: int
    action_id: str
    action_name: str
    approved: bool
    reason: str
    kernel_reasons: List[str] = field(default_factory=list)
    monitor_reason: str = ""
    # Best-effort margins / metadata (optional)
    cbf_margin: Optional[float] = None
    ifc_ok: Optional[bool] = None
    hjb_ok: Optional[bool] = None
    prompt_tokens_est: int = 0
    output_tokens_est: int = 0


@dataclass
class ProofRecord:
    version: str
    created_at: float
    goal: str
    budget: float
    trace_hash_head: str
    steps: List[ProofStep]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["record_hash"] = _sha256(json.dumps(payload, sort_keys=True, default=str))
        return json.dumps(payload, indent=2, sort_keys=True, default=str)


def write_proof(path: str, record: ProofRecord) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(record.to_json())
        f.write("\n")
