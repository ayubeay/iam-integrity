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

/// Raw 256-bit hash of the workload source as four words.
///
/// Native and wasm both call this, so the `implementation_hash` reported by
/// each runtime is derived from identical bytes by identical code. If the two
/// ever disagree, the wasm build is not running the same source and the
/// browser-vs-native comparison is invalid by construction.
pub fn implementation_hash_words() -> [u64; 4] {
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

    h
}

/// Hex form of `implementation_hash_words`.
pub fn implementation_hash() -> String {
    let mut s = String::with_capacity(64);
    for w in implementation_hash_words().iter() {
        s.push_str(&format!("{:016x}", w));
    }
    s
}

/// wasm32 export surface.
///
/// Deliberately raw C-ABI, not wasm-bindgen. The crate stays zero-dependency,
/// the timed region contains no framework glue, and the JS harness calls the
/// same `workload::run_chain` the native binary calls. `usize` is 32-bit here,
/// which is exactly why `workload.rs` keeps its index math in `u64` and narrows
/// only at the subscript.
///
/// Memory model: JS calls `fcb_alloc` once, `fcb_init` before every timed run
/// (init is excluded from timing, same as native), then `fcb_run`, then
/// `fcb_free`. `u64` parameters arrive from JS as BigInt.
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use crate::workload::{self, Params};

    /// Allocate a `words`-long scratchpad in linear memory. Returns the base
    /// pointer, or null on allocation failure. Caller must `fcb_free` it.
    #[no_mangle]
    pub extern "C" fn fcb_alloc(words: u32) -> *mut u64 {
        let mut v = vec![0u64; words as usize];
        let p = v.as_mut_ptr();
        core::mem::forget(v);
        p
    }

    /// # Safety
    /// `ptr`/`words` must be a pair returned by a prior `fcb_alloc`.
    #[no_mangle]
    pub unsafe extern "C" fn fcb_free(ptr: *mut u64, words: u32) {
        drop(Vec::from_raw_parts(ptr, words as usize, words as usize));
    }

    /// # Safety
    /// `ptr` must reference `words` valid `u64` slots.
    #[no_mangle]
    pub unsafe extern "C" fn fcb_init(ptr: *mut u64, words: u32, seed: u64) {
        let pad = core::slice::from_raw_parts_mut(ptr, words as usize);
        workload::init_scratchpad(pad, seed);
    }

    /// Run the timed chain. Writes the 4-word digest to `out`.
    ///
    /// # Safety
    /// `ptr` must reference `words` valid slots; `out` must reference 4 slots.
    #[no_mangle]
    pub unsafe extern "C" fn fcb_run(
        ptr: *mut u64,
        words: u32,
        steps: u64,
        seed: u64,
        out: *mut u64,
    ) {
        let pad = core::slice::from_raw_parts_mut(ptr, words as usize);
        let params = Params { seed, scratchpad_words: words as u64, steps };
        let d = workload::run_chain(pad, params);
        core::slice::from_raw_parts_mut(out, 4).copy_from_slice(&d.0);
    }

    /// Write the 4-word implementation hash to `out`. Must equal the value the
    /// native binary prints, or the two are not running the same source.
    ///
    /// # Safety
    /// `out` must reference 4 valid `u64` slots.
    #[no_mangle]
    pub unsafe extern "C" fn fcb_impl_hash(out: *mut u64) {
        let w = crate::implementation_hash_words();
        core::slice::from_raw_parts_mut(out, 4).copy_from_slice(&w);
    }
}
