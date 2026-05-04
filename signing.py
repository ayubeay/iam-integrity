"""
Cryptographic signing service for the IAM platform.

Provides canonical JSON serialization, SHA-256 hashing, and Ed25519
signing primitives used to produce verifiable receipts. Used by:
  - SURVIVOR challenge settlement receipts (verity_api_gated.py)
  - Agent birth receipts (mint_agent.py)
  - Future: integrity trail entries, scope amendments, etc.

Configuration:
  Set VYRE_SIGNING_SEED_HEX env var to a 32-byte hex seed. If absent,
  signing is unavailable and sign_receipt() returns (None, None, None).
  Callers must handle the unavailable case explicitly.

Public surface:
  canonical_json(obj)         -> str
  hash_receipt(canonical_str) -> str
  sign_receipt(canonical_str) -> (sig_hex, signer_key_id, verify_key_hex)
  get_verify_key_hex()        -> Optional[str]
  is_signing_configured()     -> bool
  is_nacl_available()         -> bool
  SIGNER_KEY_ID               -> str constant ("vyre_v1")
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Tuple

# ── nacl availability ────────────────────────────────────────────────────────

try:
    import nacl.signing
    import nacl.encoding
    _NACL_AVAILABLE = True
except ImportError:
    _NACL_AVAILABLE = False

# ── Public constants ─────────────────────────────────────────────────────────

SIGNER_KEY_ID = "vyre_v1"

# ── Internal state ───────────────────────────────────────────────────────────

_signing_key = None
_verify_key_hex: Optional[str] = None


def _init_signing_key():
    """
    Initialize Ed25519 signing key from VYRE_SIGNING_SEED_HEX env var.
    Idempotent — safe to call multiple times.
    Silently no-ops if nacl is unavailable or seed is missing/invalid.
    """
    global _signing_key, _verify_key_hex

    if not _NACL_AVAILABLE:
        return

    seed_hex = os.environ.get("VYRE_SIGNING_SEED_HEX", "").strip()
    if not seed_hex:
        return

    try:
        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError("VYRE_SIGNING_SEED_HEX must decode to exactly 32 bytes")
        _signing_key = nacl.signing.SigningKey(seed)
        _verify_key_hex = _signing_key.verify_key.encode(
            encoder=nacl.encoding.HexEncoder
        ).decode()
    except Exception as e:
        print(f"[signing] init failed: {e}")
        _signing_key = None
        _verify_key_hex = None


# Initialize on import
_init_signing_key()


# ── Public API ───────────────────────────────────────────────────────────────

def canonical_json(obj: dict) -> str:
    """
    Produce canonical JSON: sorted keys, no whitespace, UTF-8.
    Two semantically equal payloads produce identical strings,
    enabling deterministic hashing and signing.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_receipt(canonical: str) -> str:
    """SHA-256 hash of a canonical JSON string. Returns 'sha256:<hex>'."""
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sign_receipt(canonical: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Sign canonical JSON with the configured Ed25519 key.
    Returns (signature_hex, signer_key_id, verify_key_hex).
    Returns (None, None, None) if signing is unavailable.
    Callers must check the return values before using them.
    """
    if not _NACL_AVAILABLE or _signing_key is None:
        return (None, None, None)

    signed = _signing_key.sign(canonical.encode("utf-8"))
    sig_hex = signed.signature.hex()
    return (sig_hex, SIGNER_KEY_ID, _verify_key_hex)


def get_verify_key_hex() -> Optional[str]:
    """Return the hex-encoded public verify key, or None if signing isn't configured."""
    return _verify_key_hex


def is_signing_configured() -> bool:
    """True if a signing key has been successfully initialized."""
    return _signing_key is not None


def is_nacl_available() -> bool:
    """True if the nacl library is importable in this environment."""
    return _NACL_AVAILABLE


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Local self-test:
        VYRE_SIGNING_SEED_HEX=$(python3 -c "import os; print(os.urandom(32).hex())") \
            python3 signing.py
    """
    print("=== signing.py self-test ===\n")
    print(f"nacl_available:       {is_nacl_available()}")
    print(f"signing_configured:   {is_signing_configured()}")
    print(f"verify_key_hex:       {get_verify_key_hex()}")
    print(f"SIGNER_KEY_ID:        {SIGNER_KEY_ID}")
    print()

    # canonical_json determinism check
    obj_a = {"b": 2, "a": 1, "c": [3, 1, 2]}
    obj_b = {"a": 1, "c": [3, 1, 2], "b": 2}
    canon_a = canonical_json(obj_a)
    canon_b = canonical_json(obj_b)
    assert canon_a == canon_b, "canonical_json non-deterministic"
    print(f"canonical_json:       {canon_a}")
    print(f"deterministic:        OK (same output for equivalent dicts)")
    print()

    # hash check
    h = hash_receipt(canon_a)
    print(f"hash_receipt:         {h}")
    assert h.startswith("sha256:"), "hash format wrong"
    assert len(h) == 7 + 64, "hash length wrong"
    print(f"hash format:          OK")
    print()

    # sign check
    sig, signer, vk = sign_receipt(canon_a)
    if sig is None:
        print("sign_receipt:         (skipped — signing not configured)")
        print()
        print("To test signing, set VYRE_SIGNING_SEED_HEX:")
        print('  export VYRE_SIGNING_SEED_HEX=$(python3 -c "import os; print(os.urandom(32).hex())")')
    else:
        print(f"sign_receipt:")
        print(f"  signature:          {sig[:32]}...")
        print(f"  signer_key_id:      {signer}")
        print(f"  verify_key_hex:     {vk}")
        assert len(sig) == 128, "Ed25519 sig should be 64 bytes / 128 hex chars"
        print(f"signature length:     OK ({len(sig)} hex chars = 64 bytes)")

        # Verify round-trip
        try:
            import nacl.signing
            import nacl.encoding
            verify_key = nacl.signing.VerifyKey(vk, encoder=nacl.encoding.HexEncoder)
            verify_key.verify(canon_a.encode("utf-8"), bytes.fromhex(sig))
            print(f"signature verifies:   OK (round-trip valid)")
        except Exception as e:
            print(f"signature verifies:   FAIL ({e})")

    print("\nAll signing.py self-tests passed.")
