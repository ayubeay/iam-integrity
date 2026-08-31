# External Collaboration Register

Live operational record. Governance: [`README.md`](./README.md).
Last reviewed: 2026-08-31.

**Next-action summary**

    WAITING_ON_US / READY_TO_RUN ....... 5
    AWAITING_COUNTERPARTY .............. 2
    PROPOSED (lead only) ............... 1
    UNRESOLVED VERIFICATION ............ 1  (see final section)

---

## Finance__broski

    platform             Reddit
    canonical spec       docs/research/experiments/EXP-PERSISTENCE-001.md
    question             Does a latent-state/hazard formulation beat a recent
                         hedge-ratio-drift baseline at predicting pair-relationship
                         persistence, out of sample?
    their contribution   drift baseline established on their universe; target and
                         censoring logic
    our commitment       reproduce the baseline SHAPE on our universe — not their
                         decimals — document the result even if negative, then test
                         latent-state/hazard
    evidence boundary    our universe and windows only; P&L excluded entirely
    IP boundary          their protocol logic is theirs; we reproduce structure, not code
    next-action owner    US
    next action          reproduce drift baseline
    last interaction     2026-08
    status               READY_TO_RUN
    result               —

    note  A null result is valid and the collaborator has explicitly asked to be told.

---

## Purple-Ad6867

    platform             Reddit
    canonical spec       docs/research/experiments/EXP-SELECTION-001.md · Case 001
    question             Can an internal prediction receipt be reconciled against an
                         independent external pre-outcome record?
    their contribution   opened their prediction board for the experiment
    our commitment       use an existing model with existing rules; submit only naturally
                         qualifying predictions; abstain otherwise; begin manually
    evidence boundary    the specific ledger and board window; says nothing about
                         predictive skill
    IP boundary          board data is theirs; we publish our reconciliation, not their
                         participants' records
    next-action owner    US
    next action          small manual prediction sample
    last interaction     2026-08
    status               READY_TO_RUN
    result               —

    constraint  Do not tune the model toward the board. Abstention is data, not failure;
                coverage and cadence are measured separately from accuracy.

---

## One_Weather_9417 / Leah

    platform             Reddit
    canonical spec       UNRESOLVED — no falsifiable protocol yet
    question             What does ~6 months of testing and CISO discussion actually
                         demonstrate about social-engineering intervention, and what
                         remains hypothesis?
    their contribution   promised testing evidence and CISO feedback
    our commitment       on receipt, separate DEMONSTRATED from REMAINING HYPOTHESIS
                         from BUYER WILLINGNESS before proposing any product form
    evidence boundary    pending
    IP boundary          their client/CISO material is confidential; no named parties
                         enter our records
    next-action owner    COUNTERPARTY
    next action          receive promised material
    last interaction     2026-08 — "before end of this week"
    status               AWAITING_COUNTERPARTY
    result               —

    constraint  Do not assume the workshop is the product. Do not construct a
                commercialization map before the evidence arrives.

---

## Various_Payment_7956

    platform             Reddit
    canonical spec       UNRESOLVED — facility discovery is not a falsifiable protocol
    question             Can an existing facility execute a bounded preliminary protocol,
                         instead of ~$60-70K for a custom 1-2 L prototype with joint-IP
                         implications?
    their contribution   reduced the immediate blocker to high-pressure + supercritical
                         CO2 equipment access
    our commitment       investigate US university, shared, contract or fee-for-service
                         facilities capable of supporting a bounded experiment
    evidence boundary    capability fit only
    IP boundary          do NOT request proprietary formulations, operating parameters or
                         IP-sensitive chemistry; seek capability fit first
    next-action owner    US
    next action          facility discovery
    last interaction     2026-08-29
    status               WAITING_ON_US
    result               —

    doctrine  Do not route a founder to capital before diagnosing whether the underlying
              bottleneck is really capital. Here it appears to be access, not funding.

---

## RemoteStreet815 / Opta

    platform             Reddit → direct → Telegram private beta
    canonical spec       UNRESOLVED — generalizable findings may inform existing
                         execution/risk/admissibility doctrine later
    question             How does the protocol actually behave at its specification
                         boundaries — payoff cap, expiry window, settlement recovery,
                         dependency degradation?
    their contribution   functioning external protocol, devnet environment, whitepaper,
                         founder access
    our commitment       execution-path analysis, edge-case testing, spec-vs-runtime
                         reconciliation, structured feedback
    evidence boundary    devnet only; their implementation, not ours
    IP boundary          STRICT BOTH WAYS. Opta protocol specifics and IP do not enter
                         canon. Our internal architecture is not disclosed to them.
    next-action owner    US
    next action          expiry/settlement replay on devnet
    last interaction     2026-08 — a long question was removed by group moderation;
                         founder asked us to repost and tag
    status               ACTIVE
    result               —

    open threads  (1) payoff/pricing reconciliation — whether the capped-call payoff
                  min(max(S-K,0),K) is reflected in the stated pricing model;
                  (2) ±60s settlement window vs ~5-minute crank cadence — deliberately
                  miss the window on devnet and observe actual recovery semantics rather
                  than infer them from documentation;
                  (3) dependency incident behaviour — pricing paused after an upstream
                  provider reported a security incident; which state froze, which
                  continued, which obligations still executed.

    note  The moderation removal and repost request are COLLABORATION STATE, not research
          evidence. They belong here and not in any finding.

---

## Altruistic-Leave-359 / Delta Hive

    platform             Reddit → Discord
    canonical spec       UNRESOLVED — adjacent to
                         docs/research/experiments/EXP-GENEALOGY-001.md but not the same
                         object: that spec concerns evidence ancestry in agent belief;
                         this concerns opportunity clustering in markets
    question             Do seven bots represent seven independent market opportunities,
                         or fewer economic episodes responding to shared underlying moves?
    their claim          the seven bots were deliberately designed independent — distinct
                         parameters, triggers and "DNA"
    our commitment       characterize what the environment legitimately exposes, then
                         determine whether a useful experiment is possible without
                         requiring proprietary strategy logic
    evidence boundary    observation only
    IP boundary          do NOT request strategy logic. Posted P&L is SELF_REPORTED,
                         never INDEPENDENTLY_VERIFIED.
    next-action owner    US
    next action          characterize legitimately observable data
    last interaction     2026-08
    status               WAITING_ON_US
    result               —

    distinction  Design independence is not statistical independence is not economic
                 opportunity independence. Their claim addresses the first only.

    constraint   Do not message the collaborator again before determining what the
                 environment exposes.

---

## JadedHome6490

    platform             Reddit
    canonical spec       none — no commitment on either side
    question             —
    our commitment       none
    next-action owner    US
    next action          decide whether a falsifiable protocol exists around state
                         management, tool execution, retries, approvals, multi-agent
                         coordination, model routing, observability or runtime economics
    last interaction     2026-08
    status               PROPOSED
    result               —

    note  Architecture comparison and discovery. Recorded as a lead so it is not
          mistaken for an experiment, and not lost.

---

## Bantex29 / VectorStep

    platform             UNRECORDED — platform not stated in intake
    canonical spec       none — the central architectural distinction is already
                        owned by docs/reserve/computable-accountability.md; no
                        experiment protocol agreed
    question             Where does observability / traceability end and operational
                        accountability begin, when telemetry transport exists but
                        identity, policy, authority, admissibility and responsibility
                        semantics remain separate?
    their contribution   response pending
    our commitment       preserve their response as stated, including disagreement
                        with our framing; do not normalize it into existing
                        architecture merely to make it fit
    evidence boundary    pending — no response received and nothing independently
                        reproduced
    IP boundary          their implementation and material remain theirs; no
                        proprietary observability or runtime design enters canon;
                        our internal architecture and methodology are not disclosed
                        beyond what the conversation requires
    next-action owner    COUNTERPARTY
    next action          receive response to the probe
    last interaction     2026-08 — probe sent
    status               AWAITING_COUNTERPARTY
    result               —

    context  Counterparty described as an observability engineer associated with
             VectorStep. That is counterparty-reported context, not verified
             capability, and no claim of theirs has been independently reproduced.

    note  The distinction prompting the probe already collides with
          computable-accountability.md, which separates telemetry, interpretation,
          recommendation, admissibility, authority, execution and receipts. This
          record does not establish that the counterparty agrees with, validates or
          disproves that doctrine — no response has arrived. A counterparty response
          is not canonical validation, and self-reported material is never VERIFIED.

---

## Unresolved verification — not a collaboration

## AICryptocomp / AIcryptott1

    platform             GitHub organization invitation
    canonical spec       n/a — this is not a research collaboration
    status               UNRESOLVED VERIFICATION · invitation deliberately left PENDING
    next-action owner    COUNTERPARTY
    last interaction     2026-08 — verification challenge sent, unanswered

    established          org created 2026-08-24; account 95 minutes old at inspection;
                         target repository returned 404; recruitment activity appears
                         publicly corroborated
    NOT established      employer/company identity; GitHub provenance; whether the
                         repository exists

    standing constraints  Do not accept the invitation while verification is unresolved.
                          Do not disclose inspection methodology, commands run, token
                          scopes, local repository inventory, or any internal architecture.
                          Repository inspection is not satisfaction of the unanswered trust
                          questions and is not a commitment to proceed.

    note  Recorded here because a pending invitation with an unanswered challenge is
          exactly the dangling external state this register exists to keep visible.
