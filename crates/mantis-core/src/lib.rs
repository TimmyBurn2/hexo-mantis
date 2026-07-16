//! mantis-core: board, hex geometry, rules, Ply/Turn vocabulary types.
//! WP0 scaffold: compiles empty; board port lands with the core work package.

pub const CRATE_NAME: &str = "mantis-core";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_pinned() {
        assert_eq!(super::CRATE_NAME, "mantis-core");
    }
}
