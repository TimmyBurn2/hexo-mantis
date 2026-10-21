//! The bindgen attribute lives in `mantis-player/src/wasm.rs` alone, and that module builds for wasm32 only.

use std::path::Path;

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("a readable source tree") {
        let path = entry.expect("a readable entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn the_wasm_bindings_live_in_the_players_wasm_module_alone() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let wasm = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/wasm.rs");
    let mut files = Vec::new();
    rust_files(&crates, &mut files);
    let marker = concat!("#[wasm", "_bindgen");
    let holders: Vec<_> = files
        .iter()
        .filter(|f| std::fs::read_to_string(f).is_ok_and(|s| s.contains(marker)))
        .filter_map(|f| f.canonicalize().ok())
        .collect();
    assert_eq!(holders, vec![wasm.canonicalize().expect("wasm.rs exists")]);
    let lib = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("lib.rs");
    assert!(
        lib.contains("#[cfg(target_arch = \"wasm32\")]\npub mod wasm;"),
        "wasm.rs is gated to wasm32"
    );
}
