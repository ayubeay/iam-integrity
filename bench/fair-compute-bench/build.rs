//! Captures the toolchain and target at build time so every native result
//! records how it was produced. No runtime cost, no dependency — the strings
//! are baked in as environment variables the binary reads via env!().
//!
//! This is the reproducibility metadata the reserve calls for: a result is only
//! comparable across machines if you know the compiler and target that made it.

use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=FCB_RUSTC_VERSION={}", version);

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=FCB_TARGET={}", target);

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=FCB_PROFILE={}", profile);
}
