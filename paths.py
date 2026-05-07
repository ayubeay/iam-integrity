"""
Path configuration for the IAM-integrity service.

Two roots, one rule:
  CODE_ROOT — where source files live (always working directory)
  DATA_ROOT — where mutable runtime state lives
              local default: same as CODE_ROOT (current behavior preserved)
              production:    /data (Railway volume mount, persists across deploys)

Critical rule:
  STATIC templates (committed, ship with code)   route through CODE_ROOT
  MUTABLE state (registry, trail, scopes, oras)  route through DATA_ROOT

Set IAM_DATA_ROOT in production to point mutable paths at a persistent
volume mount. Without that, every Railway deploy overwrites the registry.

Static (CODE_ROOT):
  archetypes.json
  agents_zodiac.json
  data/sonic_catalog_seed.json
  challenges/

Mutable (DATA_ROOT):
  agents_index.json
  agents_seed.json           (rebuilt by verity_indexer.py)
  integrity_trail.jsonl
  scopes/                    (per-agent scope contracts)
  oras/                      (governance contracts, including bootstrapped defaults)
  data/survivor_data.json    (SURVIVOR staking state)
  data/helixcan_snapshot.json
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Roots ────────────────────────────────────────────────────────────────────

CODE_ROOT: Path = Path(__file__).resolve().parent

DATA_ROOT: Path = Path(os.environ.get("IAM_DATA_ROOT", str(CODE_ROOT))).resolve()

# ── Static templates (always code root) ──────────────────────────────────────

ARCHETYPES_PATH:    Path = CODE_ROOT / "archetypes.json"
AGENTS_ZODIAC_PATH: Path = CODE_ROOT / "agents_zodiac.json"
SONIC_CATALOG_PATH: Path = CODE_ROOT / "data" / "sonic_catalog_seed.json"
CHALLENGES_DIR:     Path = CODE_ROOT / "challenges"

# ── Mutable runtime state (data root) ────────────────────────────────────────

AGENTS_INDEX_PATH:      Path = DATA_ROOT / "agents_index.json"
AGENTS_SEED_PATH:       Path = DATA_ROOT / "agents_seed.json"
INTEGRITY_TRAIL_PATH:   Path = DATA_ROOT / "integrity_trail.jsonl"
SCOPES_DIR:             Path = DATA_ROOT / "scopes"
ORAS_DIR:               Path = DATA_ROOT / "oras"
SURVIVOR_DATA_FILE:     Path = DATA_ROOT / "data" / "survivor_data.json"
HELIXCAN_SNAPSHOT_PATH: Path = DATA_ROOT / "data" / "helixcan_snapshot.json"


def ensure_runtime_dirs() -> None:
    """
    Create runtime directories if they don't exist.
    Idempotent — safe to call on every startup.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SCOPES_DIR.mkdir(parents=True, exist_ok=True)
    ORAS_DIR.mkdir(parents=True, exist_ok=True)
    SURVIVOR_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)


def describe() -> dict:
    """Return current path configuration as a dict (for debugging/logging)."""
    return {
        "CODE_ROOT": str(CODE_ROOT),
        "DATA_ROOT": str(DATA_ROOT),
        "data_root_separate": str(CODE_ROOT) != str(DATA_ROOT),
        "iam_data_root_env": os.environ.get("IAM_DATA_ROOT", "(unset — using CODE_ROOT)"),
        "static": {
            "archetypes":    str(ARCHETYPES_PATH),
            "zodiac":        str(AGENTS_ZODIAC_PATH),
            "sonic_catalog": str(SONIC_CATALOG_PATH),
            "challenges":    str(CHALLENGES_DIR),
        },
        "mutable": {
            "agents_index":     str(AGENTS_INDEX_PATH),
            "agents_seed":      str(AGENTS_SEED_PATH),
            "integrity_trail":  str(INTEGRITY_TRAIL_PATH),
            "scopes_dir":       str(SCOPES_DIR),
            "oras_dir":         str(ORAS_DIR),
            "survivor_data":    str(SURVIVOR_DATA_FILE),
            "helixcan_snap":    str(HELIXCAN_SNAPSHOT_PATH),
        },
    }


if __name__ == "__main__":
    import json
    print("=== paths.py configuration ===\n")
    print(json.dumps(describe(), indent=2))
