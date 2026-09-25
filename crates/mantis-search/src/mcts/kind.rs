//! The search a tree runs, as ONE closed kind.

/// Which search this tree runs — PUCT, or Gumbel-Top-k with Sequential Halving.
///
/// ONE closed key, replacing four switches (`gumbel_mcts`, `gumbel_variant`, and a
/// `completed_q_values` flag in each of selfplay and train): the corrections that make Gumbel
/// correct are what corrected Gumbel IS, so they travel as a kind.
///
/// NO `Default`, deliberately: an absent search kind is a config-seam error, never a silent PUCT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchKind {
    /// PUCT descent everywhere, Dirichlet root noise, visit-count targets.
    Puct,
    /// Gumbel-Top-k root sampling with Sequential Halving, completed-Q interior
    /// selection and completed-Q improved-policy targets. Follows
    /// `google-deepmind/mctx`'s `gumbel_muzero_policy`.
    Gumbel,
}

impl SearchKind {
    /// Parse the config spelling. `None` for anything else — the caller decides whether
    /// an unknown kind is a boot error or a test typo. There is no fallback arm.
    #[must_use]
    pub fn from_config_str(s: &str) -> Option<Self> {
        match s {
            "puct" => Some(SearchKind::Puct),
            "gumbel" => Some(SearchKind::Gumbel),
            _ => None,
        }
    }

    /// The config spelling, so a caller never re-types the literal.
    #[must_use]
    pub fn as_config_str(self) -> &'static str {
        match self {
            SearchKind::Puct => "puct",
            SearchKind::Gumbel => "gumbel",
        }
    }

    /// Whether this kind exports the completed-Q improved policy as its training
    /// target; the one authority, so the target cannot disagree with the search.
    #[must_use]
    pub fn completed_q_target(self) -> bool {
        matches!(self, SearchKind::Gumbel)
    }

    /// Whether this kind's graph rows are stored SPARSE: the m visited candidates'
    /// exact entries plus one tail mass α, rather than a slot per legal action.
    ///
    /// Same predicate as `completed_q_target` today but a separate question: that one says what
    /// the target MEANS, this one how the row is LAID OUT.
    #[must_use]
    pub fn stores_sparse_rows(self) -> bool {
        matches!(self, SearchKind::Gumbel)
    }
}

#[cfg(test)]
mod tests {
    use super::SearchKind;

    #[test]
    fn only_the_gumbel_kind_stores_a_sparse_row() {
        assert!(SearchKind::Gumbel.stores_sparse_rows());
        assert!(!SearchKind::Puct.stores_sparse_rows());
    }
}
