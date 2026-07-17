# Transition Intelligence — Adversarial Verification

**Date:** July 17, 2026 · **Engine:** scoreTransition as shipped in commit 6319988 (extracted verbatim, run in Node)
**Result: PASSED (17/18; the single flagged case was a test-fixture error, engine judgment correct)**

## Cases
1. BAD (peak dancehall high 2010s → warm-up hip-hop low 2000s): 30% RED. Warnings: energy drops too quickly, era jump, set position backward, genre jump. BPM gap of 5 correctly judged "stretch — pitch ride needed" rather than mismatch (engine more DJ-accurate than the test expected).
2. GOOD (afrobeats medium warm-up → same profile): 97% GREEN, every cited factor matches metadata verbatim. Separation good-vs-bad: 67 points.
3. BRIDGE REPAIR (peak dancehall → afrobeats warm-up, 62% direct): via dancehall-build bridge → 72%/75% legs, avg 74. Repair works.
4. SENSITIVITY: reversed pair scores differently (100% forward build vs 78% backward); energy direction and set position drive it.
5. MISSING DATA: track without bpm_range still scores (renormalized weights); zero BPM claims emitted. No harmonic/Camelot/community claims anywhere — that data does not exist and the engine never pretends it does.

## Standing rule
A plausible explanation that is factually wrong is worse than no explanation. Every factor string must trace to actual metadata values. New factors (Camelot, community signals) join the engine only when their data sources exist.
