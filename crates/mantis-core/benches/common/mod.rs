use mantis_core::board::Board;

/// A deterministic, collision-free board: the lexicographic-minimum legal move each ply.
pub fn board_with_n_stones(n_stones: usize) -> Board {
    let mut b = Board::new();
    for _ in 0..n_stones {
        let mv = *b.legal_moves_set().iter().min().expect("no legal moves");
        b.apply_move(mv.0, mv.1).expect("apply failed");
    }
    b
}
