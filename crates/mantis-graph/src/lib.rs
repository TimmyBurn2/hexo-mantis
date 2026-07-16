//! mantis-graph: dep-free axis-graph builder; native + wasm32 targets; MUST stay
//! dependency-free — the wasm gate checks this crate alone.
//! WP0 scaffold: compiles empty; graph builder port lands with its work package.

pub const CRATE_NAME: &str = "mantis-graph";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_pinned() {
        assert_eq!(super::CRATE_NAME, "mantis-graph");
    }
}
