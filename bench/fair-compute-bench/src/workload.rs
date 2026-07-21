//! Deterministic dependent-memory workload.
//!
//! DESIGN CONSTRAINTS — do not violate these without re-deriving the KAT vectors.
//!
//! 1. Integer-only. No floating point anywhere. f64 rounding is specified by
//!    IEEE-754 and is in practice identical, but fused-multiply-add contraction
//!    and x87 80-bit intermediates are NOT portable. Integers are.
//! 2. All arithmetic is `u64` with explicit wrapping. `usize` is 64-bit on
//!    native targets and 32-bit on wasm32 — indexing math must therefore be done
//!    in u64 and narrowed to usize only at the final subscript.
//! 3. The access chain is strictly serial: the address of step N+1 cannot be
//!    computed until the value read at step N has been mixed into the
//!    accumulator. This is the whole point. It makes the workload bound by
//!    dependent memory latency, not by bandwidth or by parallel throughput.
//! 4. Every step writes back to the slot it read. A read-only chain can be
//!    served from a replicated read-only cache; a read-modify-write chain
//!    cannot, and it defeats speculative prefetch.
//! 5. No threads, no allocation inside the hot loop, no syscalls, no I/O.
//!
//! None of this makes the workload *fair*. It makes it *measurable*. Whether
//! the fairness thesis survives is an empirical question this crate exists to
//! answer, and the honest expected outcome is that it may not.

/// 256-bit workload output. Two runs with identical parameters must produce
/// identical digests on every implementation and every target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Digest(pub [u64; 4]);

impl Digest {
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for w in self.0.iter() {
            s.push_str(&format!("{:016x}", w));
        }
        s
    }
}

/// Workload parameters. `scratchpad_words` must be a power of two so the index
/// mask is exact; a modulo would introduce a division into the dependent chain
/// and would measure the divider rather than the memory system.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub seed: u64,
    pub scratchpad_words: u64,
    pub steps: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParamError {
    NotPowerOfTwo(u64),
    TooSmall(u64),
    TooLarge(u64),
    ZeroSteps,
}

impl core::fmt::Display for ParamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParamError::NotPowerOfTwo(n) => {
                write!(f, "scratchpad_words must be a power of two, got {}", n)
            }
            ParamError::TooSmall(n) => write!(
                f,
                "scratchpad_words must be >= {}, got {}",
                MIN_WORDS, n
            ),
            ParamError::TooLarge(n) => write!(
                f,
                "scratchpad_words must be <= {} (wasm32 addressability), got {}",
                MAX_WORDS, n
            ),
            ParamError::ZeroSteps => write!(f, "steps must be > 0"),
        }
    }
}

/// 8 KiB. Below this the whole scratchpad lives in L1 and the benchmark
/// measures cache hit latency, which is not the quantity of interest.
pub const MIN_WORDS: u64 = 1 << 10;

/// 2^29 words = 4 GiB. wasm32 cannot address beyond 4 GiB, and holding the
/// native ceiling to the wasm ceiling keeps the two comparable by construction.
pub const MAX_WORDS: u64 = 1 << 29;

impl Params {
    pub fn validate(&self) -> Result<(), ParamError> {
        if self.steps == 0 {
            return Err(ParamError::ZeroSteps);
        }
        if !self.scratchpad_words.is_power_of_two() {
            return Err(ParamError::NotPowerOfTwo(self.scratchpad_words));
        }
        if self.scratchpad_words < MIN_WORDS {
            return Err(ParamError::TooSmall(self.scratchpad_words));
        }
        if self.scratchpad_words > MAX_WORDS {
            return Err(ParamError::TooLarge(self.scratchpad_words));
        }
        Ok(())
    }

    pub fn scratchpad_bytes(&self) -> u64 {
        self.scratchpad_words * 8
    }
}

/// splitmix64. Chosen because it is short, has no lookup tables (a table would
/// add a second, competing memory access pattern), passes BigCrush, and is
/// trivially reimplementable in any language — which matters, because the
/// cross-implementation determinism check is only meaningful if a second
/// implementation is cheap to write.
#[inline(always)]
pub fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fill the scratchpad deterministically. This is a bandwidth-bound streaming
/// write and is deliberately NOT part of the timed region — including it would
/// contaminate a latency measurement with a bandwidth measurement.
pub fn init_scratchpad(pad: &mut [u64], seed: u64) {
    for (i, slot) in pad.iter_mut().enumerate() {
        *slot = mix(seed ^ (i as u64).wrapping_mul(0x1000_0000_0000_01B3));
    }
}

/// Result of the timed dependent-access loop, before digest finalization.
#[derive(Clone, Copy, Debug)]
pub struct LoopState {
    /// Final accumulator, fed into `finalize`.
    pub acc: u64,
    /// Number of iterations actually executed. This is returned by the loop
    /// itself, not taken on faith from the caller's argument, so a harness can
    /// assert `executed_steps == requested_steps` independently of the digest.
    pub executed_steps: u64,
}

/// The timed region — the dependent-access loop ONLY.
///
/// Digest finalization is deliberately NOT here: it is a strided fold whose cost
/// is bandwidth, not dependent latency, and mixing it into the timed number
/// contaminates the measurement. Time this; call `finalize` after the clock
/// stops.
///
/// The caller MUST consume the returned state (the harness passes it through a
/// black-box). This loop is pure computation over a buffer the caller owns; if
/// the result is discarded, the optimiser is entitled to delete it and the
/// benchmark measures nothing.
pub fn run_loop(pad: &mut [u64], params: Params) -> LoopState {
    let mask = params.scratchpad_words - 1;
    let mut acc = mix(params.seed ^ 0xA5A5_5A5A_C3C3_3C3C);

    let mut step: u64 = 0;
    while step < params.steps {
        // The mask is exact because scratchpad_words is a power of two.
        // Narrowing to usize is safe: mask < MAX_WORDS <= u32::MAX on wasm32.
        let idx = (acc & mask) as usize;
        let v = pad[idx];
        acc = mix(acc ^ v).wrapping_add(step);
        pad[idx] = v ^ acc;
        step = step.wrapping_add(1);
    }

    LoopState { acc, executed_steps: step }
}

/// Loop plus digest. Behaviour is identical to the original `run_chain`; the KAT
/// suite pins its output, so the loop/finalize split above cannot silently
/// change semantics.
pub fn run_chain(pad: &mut [u64], params: Params) -> Digest {
    let state = run_loop(pad, params);
    finalize(pad, state.acc, params)
}

/// The digest covers the accumulator (which every step feeds) and a strided
/// sample of the scratchpad (which catches a write-back path that silently
/// stops working). A full fold over the pad would be bandwidth work and would
/// dominate short runs. Called after the clock stops — not part of the timed
/// loop.
pub fn finalize(pad: &[u64], acc: u64, params: Params) -> Digest {
    let mut h0 = mix(acc);
    let mut h1 = mix(acc ^ params.steps);
    let mut h2 = mix(acc ^ params.scratchpad_words);
    let mut h3 = mix(acc ^ params.seed);

    const SAMPLES: u64 = 1024;
    let stride = if params.scratchpad_words > SAMPLES {
        params.scratchpad_words / SAMPLES
    } else {
        1
    };

    let mut i: u64 = 0;
    while i < params.scratchpad_words {
        let v = pad[i as usize];
        h0 = mix(h0 ^ v);
        h1 = h1.wrapping_add(mix(v ^ i));
        h2 ^= mix(h2 ^ v);
        h3 = mix(h3.wrapping_add(v));
        i += stride;
    }

    Digest([h0, h1, h2, h3])
}

/// Allocate, initialise, and run. Returns the digest.
pub fn execute(params: Params) -> Result<Digest, ParamError> {
    params.validate()?;
    let mut pad = vec![0u64; params.scratchpad_words as usize];
    init_scratchpad(&mut pad, params.seed);
    Ok(run_chain(&mut pad, params))
}
