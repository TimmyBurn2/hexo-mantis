//! The search a tree runs, as ONE closed kind.

/// Which search this tree runs — PUCT, or Gumbel-Top-k with Sequential Halving.
///
/// ONE key, and a CLOSED one. The predecessor surface carried four independent
/// switches (`gumbel_mcts`, `gumbel_variant`, and a `completed_q_values` flag on each
/// of the self-play and train sections), which spelled sixteen regimes of which two
/// were ever run and none of the other fourteen was ever measured. The corrections
/// that made Gumbel correct are not independently meaningful — they are what corrected
/// Gumbel IS — so they travel as a kind rather than as a lattice.
///
/// NO `Default`, deliberately (LAW-11). An absent search kind is an error at the config
/// seam, never a silent PUCT; a `Default` impl here would be the code-side default that
/// makes the schema's requirement unenforceable.
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
    /// target. THE one authority: the target semantics used to be a second pair of
    /// config flags that could disagree with the search that produced them.
    #[must_use]
    pub fn completed_q_target(self) -> bool {
        matches!(self, SearchKind::Gumbel)
    }

    /// R347(a) — whether this kind's graph rows are stored SPARSE: the m visited candidates'
    /// exact entries plus one tail mass α, rather than a slot per legal action.
    ///
    /// It is the same predicate as `completed_q_target` today and is deliberately a SECOND
    /// method rather than a reuse of the first: they answer different questions (what the
    /// target MEANS versus how the row is LAID OUT), and a kind that completed its Q-values
    /// but visited every action would want the first and not the second.
    #[must_use]
    pub fn stores_sparse_rows(self) -> bool {
        matches!(self, SearchKind::Gumbel)
    }
}

#[cfg(test)]
mod tests {
    use super::SearchKind;

    #[test]
    fn the_config_spelling_round_trips() {
        for kind in [SearchKind::Puct, SearchKind::Gumbel] {
            assert_eq!(
                SearchKind::from_config_str(kind.as_config_str()),
                Some(kind),
                "{kind:?} must survive a round trip through its config spelling"
            );
        }
    }

    #[test]
    fn an_unknown_kind_is_refused_rather_than_defaulted() {
        for s in ["", "legacy", "mctx", "PUCT", "gumbel_mcts", "true"] {
            assert!(
                SearchKind::from_config_str(s).is_none(),
                "{s:?} must not parse — a typo'd kind silently becoming PUCT is the \
                 silent-fallback class LAW-11 closes"
            );
        }
    }

    #[test]
    fn the_completed_q_target_is_the_kinds_own_answer() {
        assert!(SearchKind::Gumbel.completed_q_target());
        assert!(!SearchKind::Puct.completed_q_target());
    }

    #[test]
    fn only_the_gumbel_kind_stores_a_sparse_row() {
        assert!(SearchKind::Gumbel.stores_sparse_rows());
        assert!(!SearchKind::Puct.stores_sparse_rows());
    }
}
