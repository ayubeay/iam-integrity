//! Milestone A of the browser-fair-compute reserve: a native Rust
//! dependent-memory workload with a deterministic digest and a stable harness.
//!
//! See docs/browser-fair-compute-reserve.md for scope. Milestone B compiles
//! `workload.rs` -- unchanged -- to wasm32. Nothing in this module may acquire
//! a dependency on std beyond `alloc`/`format!`, or that becomes impossible.

pub mod workload;

/// The exact source text of the workload, embedded at compile time.
///
/// The point of hashing the source rather than the binary is that Milestone B
/// must run *the same implementation*, not a port of it. If the wasm build
/// reports a different implementation hash, the comparison is invalid and the
/// benchmark report must say so.
pub const WORKLOAD_SRC: &str = include_str!("workload.rs");

/// 256-bit hash of the workload source, using the workload's own mixing
/// function so that no external hash dependency is needed.
pub fn implementation_hash() -> String {
    let bytes = WORKLOAD_SRC.as_bytes();
    let mut h: [u64; 4] = [
        0x6A09_E667_F3BC_C908,
        0xBB67_AE85_84CA_A73B,
        0x3C6E_F372_FE94_F82B,
        0xA54F_F53A_5F1D_36F1,
    ];

    // Length-prefixed so that trailing-whitespace changes cannot collide.
    h[0] ^= bytes.len() as u64;

    for (i, chunk) in bytes.chunks(8).enumerate() {
        let mut w: u64 = 0;
        for (j, b) in chunk.iter().enumerate() {
            w |= (*b as u64) << (8 * j);
        }
        // Pad short final chunks distinguishably.
        w ^= (chunk.len() as u64) << 56;

        let k = i % 4;
        h[k] = workload::mix(h[k] ^ w);
        h[(k + 1) % 4] = h[(k + 1) % 4].wrapping_add(h[k]);
    }

    for i in 0..4 {
        h[i] = workload::mix(h[i] ^ h[(i + 3) % 4]);
    }

    let mut s = String::with_capacity(64);
    for w in h.iter() {
        s.push_str(&format!("{:016x}", w));
    }
    s
}
