# Evidence Discipline

How this portfolio decides that it knows something.

This document is **not a gate**. Nothing waits on it and nothing is blocked by it. It
records the rules that produced the results already in `docs/research/`, so that later
work — Zircon included — **inherits the discipline rather than passing through it.** A gate
is something you clear once. A discipline is something that constrains every subsequent
measurement, which is the useful property.

Most of what follows was not derived in advance. It was extracted from specific errors,
each of which is named, because a rule whose origin is recorded is harder to quietly drop
than a rule that appears as received wisdom.

---

## 1 · Three records, three questions

Established in [`README.md`](./README.md) and repeated here only as the frame:

| Record | Question | Location |
|---|---|---|
| Implementation status | What exists and runs? | `IMPLEMENTATION_STATUS.md` |
| Reserves | What may exist — decisions preserved for possible future work? | `docs/RESERVED.md`, `docs/reserve/` |
| Research | What has measurement actually taught us? | `docs/research/` |

A system appears in all three and means something different in each. Collapsing them is
how a portfolio loses track of what it knows.

---

## 2 · Experiments and tests are different questions

An **experiment** asks whether a proposition about the world is true.
A **test** asks whether a system satisfies a required behaviour.

They take separate status vocabularies because they answer separate questions, and a
shared vocabulary would let one silently stand in for the other.

    experiments:  PROPOSED · ACTIVE · VALIDATED · REJECTED · INCONCLUSIVE · BLOCKED
    tests:        RESERVED · READY · RUNNING · PASS · FAIL · BLOCKED

**A `PASS` is never evidence that an external-world hypothesis is validated.** A test
establishes that an implementation behaved as required, on the inputs used, at the commit
tested. It says nothing about whether the proposition motivating the invariant is true of
the world.

**The reverse also holds.** An experiment's `VALIDATED` does not mean any system implements
the consequence correctly. That is what a test is for.

Directories: `docs/research/experiments/` · `docs/research/tests/`, each with its own INDEX
and register.

---

## 3 · Status attaches to claims, not to runs

A single run can validate one claim while rejecting another. Forcing one global verdict
destroys information.

The canonical case is **E1, HELIX-JANUS Epoch 1**: the ledger was internally perfect and its
market evidence inadmissible. Neither fact is the verdict on the other. Recording one
overall status would have lost whichever half it did not name.

Pre-Zircon candidate 1 is written the same way on purpose — Claim A (temporal
admissibility) and Claim B (evidence-source attribution) conclude on separate lines, so one
run cannot confound them.

---

## 4 · Every conclusion carries its evidence boundary

The provenance footer records:

    source artifact → experiment date → implementation commit → evidence boundary
    → conclusion date

The **boundary** states the scope within which the conclusion holds: one machine, one
workload, one population, one provider, one retrieval architecture.

**A conclusion recorded without its boundary is not preserved, only remembered — and it will
eventually be applied where it does not hold.**

Cross-repository provenance is written `` `<repo> <path>` ``. It is a record, not a link,
and it will not resolve inside this repository. That is correct.

---

## 5 · Pre-registration

Hypotheses and acceptance conditions are written before observation.

This is not ceremony. It is the reason an unfavourable result can be accepted without the
criterion being renegotiated afterward. **E6 is the proof it works**: a criterion fixed
before observation let a rejection stand rather than becoming a discussion about whether the
criterion had been the right one.

A spec whose acceptance condition is *"the existing reserves already handle this, so create
nothing"* has been designed correctly. Rejection is preferred to inflation.

---

## 6 · Instrument discipline

The rules in this section were earned expensively. Every one of them exists because an
instrument produced a confident answer that was an artifact of the instrument.

### 6.1 · An empty result is evidence only if the instrument is proven capable of a non-empty one

Observed four times in one session:

- a keyword scan run over a `git diff` that had already aborted with `fatal: unable to read`
  — it scanned nothing and reported clean;
- a pattern testing for runtime-emitting declarations that could not match union members or
  object properties, printing nothing that was nearly read as proof;
- `find -name "*.md"` that structurally could not see reserve-index.csv;
- a shell loop over an unquoted variable that ran once against a directory named
  `"repo-a repo-b repo-c …"`, scanning zero repositories and printing an empty result.

**Every scan must carry a control that must return non-zero.** If the control fails, the
scan is void and says so, rather than reporting a comfortable zero. Where a control is
impractical, print the count of items examined — zero examined is an instrument failure, not
a clean result.

### 6.2 · Detection is not interpretation

> **A detector establishes that an observable matches a defined pattern. Separate evidence
> is required to establish what that match means.**

This is not tooling hygiene. It is an admissibility principle, and it belongs anywhere a
system converts a signal into a claim — VERITY, LITMUS, and every automated governance path
that scores, flags or classifies. A detector's output is an observation. Treating it as a
finding skips the step where meaning is established.

Three instances, all within one day:

**A pattern matched a substring of an ordinary word.** A secret scanner keyed on `sk-`
flagged `risk-proportional`, `identity-risk-signals` and `risk-screen`. Four hits, zero
credentials.

**A pattern matched a real value with the wrong meaning.** A 44-character
`SURVIVOR_TOKEN`, committed and pushed, matched a credential pattern and was escalated
toward rotation. It is a **public Solana token mint address**, published on the project's own
site, returned as `{"token": …, "chain": "solana", "platform": "pump.fun"}`, never sent in an
authorization header. Rotating it was neither possible nor desirable. What settled it was not
a better pattern but a different question: *what is this value used for?*

**A pattern matched prose describing the thing it was searching for.** A scan of a git
reflog for `push|fetch|clone|pull` returned one hit, briefly suggesting the retired store had
network history. The match was the word *fetch* inside a commit message —
`feat: live VERITY PoW fetch with real scorer output`. `FETCH_HEAD` was 0 bytes and no remote
was ever configured. The detector was reading a description of a fetch, not a record of one.

**Resolution requires a different class of evidence than detection.** How the value is
consumed. What field holds it. Whether it crosses a trust boundary. What the surrounding
structure is. A second pattern of the same kind cannot settle what the first one found.

The failure runs both directions and both are costly: a false positive spends effort
rotating what cannot be rotated; a false negative reports a clean scan over a store holding
nine private keys.

### 6.3 · A failed resolver is not failed provenance

A reference checker that cannot resolve a string has learned something about itself, not
necessarily about the reference.

### 6.4 · Verify the instrument on a known-good subject

An unreadable object store and a broken reader produce identical silence. Run the same
command against something known to work before concluding the subject is at fault.

### 6.5 · Multiplicity and selection discipline

A capable instrument can still produce misleading evidence when a system searches many
hypotheses, parameters, sources, windows, transformations or candidate relationships and
reports only the interesting result.

**The number of reported results is not the number of opportunities to find one.** A
finding selected from many chances to look does not carry the weight of the same finding
from one pre-registered test. §5 fixes the hypothesis before observation; this rule
governs how many hypotheses were entertained before one was reported.

Where multiplicity could materially affect interpretation, preserve enough of the search
process to establish the relevant comparison set: what was tested, what was selected, what
was excluded, and what was generated after observation.

**Evidence used to generate a hypothesis is not independent confirmation of it.** A
pattern discovered in a body of data is not confirmed by that same data. This does not
require a formal split for every exploratory pass. It requires that a receipt not present
discovery evidence as though it were confirmation.

A null, randomized, permutation, surrogate, holdout or other comparison may be appropriate
depending on the domain. **This rule does not mandate a statistical technique.** Where no
adequate comparison is available, that limitation belongs in the evidence boundary required
by §4 rather than being silently treated as validation.

**A valid individual test is not an admissible selected conclusion.** §6.1 asks whether the
instrument could have produced a different answer. This asks whether the process that
selected this answer from many could have produced it by chance.

Named, as §6 requires: the reconciliation that produced this rule ran many lexical probes
per pass, each carrying positive and negative controls, and never once accounted for how
many chances a hit had to appear. That establishes the process lacked the rule. **It does**
**not establish that any finding it produced was spurious** — absence of multiplicity control
is not proof that a result was chance.

---

## 7 · Finding classification

**A gate must identify the type of uncertainty it found, rather than flattening every
unresolved string into one status.** This is the operational form of §6.3 and the most
directly reusable convention in this document.

A reference-integrity gate classifies each finding as exactly one of:

| Class | Meaning | Fails the batch? |
|---|---|---|
| `RESOLVED` | resolves repo-root-relative **or** relative to the referencing file's own directory | no |
| `PLACEHOLDER` | contains `<` or `*` — a naming template or glob, not a reference | no |
| `CROSS_REPO` | repo-qualified `` `<repo> <path>` `` naming a sibling repository | no — counted and listed |
| `SELF_PREFIXED` | repo-qualified naming *this* repository; strip the prefix and resolve | only if it then fails |
| `UNRESOLVED_BATCH` | genuinely unresolvable, in a file this batch wrote | **yes** |
| `UNRESOLVED_PREEXISTING` | genuinely unresolvable, in a file this batch did not touch | no — disclosed, not merged into the batch verdict |

The last distinction is what keeps a batch verdict honest. A batch is answerable for what it
wrote. Pre-existing defects are disclosed in full and fixed in their own commit, so that
neither the batch nor the defect is misrepresented by the other.

**Measured effect.** On the 2026-08-28 batch a flat checker reported 12 `MISSING`. Classified:
2 placeholders, 7 cross-repo, 2 self-prefixed (both files present and tracked), and **1
genuine defect**. Eleven of twelve were the instrument. The one real defect —
docs/LP_SIGNAL_APPLICABILITY_RESEARCH.md written without its `survivor-oracle` prefix — was
corrected in its own commit, `010c4be`, with the batch commit `41c9727` left untouched at its
frozen scope.

**Amending a frozen batch to fix an unrelated pre-existing defect makes the historical
boundary less truthful, not more.**

---

## 8 · Doctrines earned from specific errors

Each of these replaced a claim that had been asserted, then found unsupported.

**Commit count is a proxy, not a state change.** A registry's update doctrine was claimed
broken from commit counts alone. Retracted; only the individually evidenced divergences were
kept.

**Co-location is not dependency; dependency is not ownership.** Two systems sharing
deployment history were coupled on that basis alone. Retracted.

**Mentioned repeatedly is not a reserve candidate.** A recurring term was carried toward
canonicalization on frequency. It was a personal practice, not an architecture. Removed, and
recorded in the staging ledger as explicitly outside canon so re-discovery does not
re-propose it.

**Absence in an index is not absence from the canon.** `docs/reserve/INDEX.md` is explicitly
incomplete and carries a coverage warning. Reconciliation reads files, not the index.

**A secret exposed in error is burned regardless of whether it authenticated anything.** A
value printed in plaintext was proven to have authenticated nothing — the script carrying it
died before transmission, and the service checked a different variable, header and route.
Retired anyway. Proving a specific exposure was harmless does not restore the secret.

---

## 9 · Build boundary

Nothing in `docs/research/` is implementation authorization. A documented experiment, a
promising result, and a ranked candidate are all still research. Build activation is a
separate decision made elsewhere.

**Zircon remains reserve-only.** Its own reserve records urgency LOW as a build. This
document does not change that, and completing the pre-Zircon candidates does not either —
they exist to prevent Zircon inheriting an unexamined evidence model, not to justify starting
it.
