"""Record dataclasses and error-mode normalization for Aegis-Bench.

Leakage policy (PLAN.md rule 2): the loader exposes ONLY `turns` (agent, text, step) and
`query` as model inputs. `labels` are targets. `is_injected`, `ground_truth.*`,
`metadata.num_injected_agents` and `metadata.num_agents` are never copied into `Trajectory`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ERROR_MODES = [
    "FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5",
    "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
    "FM-3.1", "FM-3.2", "FM-3.3",
]
MODE_INDEX = {m: i for i, m in enumerate(ERROR_MODES)}

_MODE_RE = re.compile(r"^(?:FM|EM)[-_ ]?(\d)\.(\d)$", re.IGNORECASE)


def normalize_mode(code: str) -> str:
    """'EM-1.1' / 'fm_1.1' -> 'FM-1.1'. Raises ValueError on unknown codes."""
    m = _MODE_RE.match(str(code).strip())
    if not m:
        raise ValueError(f"bad error mode: {code!r}")
    out = f"FM-{m.group(1)}.{m.group(2)}"
    if out not in MODE_INDEX:
        raise ValueError(f"unknown error mode: {code!r}")
    return out


@dataclass(frozen=True)
class Turn:
    step: int
    agent: str
    text: str


@dataclass
class Trajectory:
    id: str
    split: str                      # train | val | test | who_when
    framework: str
    benchmark: str
    query: str                      # also the grouping key for leakage-safe splits
    turns: list
    labels: frozenset = field(default_factory=frozenset)   # {(agent_name, FM-x.y)}

    @property
    def agents(self) -> list:
        """Distinct agent names in order of first appearance."""
        seen: dict = {}
        for t in self.turns:
            seen.setdefault(t.agent, None)
        return list(seen)

    @property
    def label_agents(self) -> frozenset:
        return frozenset(a for a, _ in self.labels)

    @property
    def label_modes(self) -> frozenset:
        return frozenset(m for _, m in self.labels)
