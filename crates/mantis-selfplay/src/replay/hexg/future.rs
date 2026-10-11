//! The HEXG v4 future-occupancy field's contract: a row's field is its own game's next placements.

use super::FUTURE_HORIZON;

/// Stones a turn places once the opening single is down: `Board::apply_move`'s hand-off.
const TURN_STONES: usize = 2;

/// Reject a future field that cannot be its game's next placements: by count, owner in turn order, cell, or `game_length`.
pub fn validate_future_stones(
    future: &[(i16, i16, i8)],
    stones: &[(i16, i16, i8)],
    ply_index: u16,
    game_length: u16,
    (current_player, moves_remaining): (i8, u8),
) -> Result<(), String> {
    let refuse = |why: String| {
        Err(format!(
            "push_graph_position: the future-occupancy field at ply {ply_index} {why} (it is the \
             next {FUTURE_HORIZON} placements of the row's own game, or fewer where that game ended)"
        ))
    };
    if future.len() > FUTURE_HORIZON {
        return refuse(format!("holds {} stones", future.len()));
    }
    let mut cells = fxhash::FxHashSet::default();
    let mover_stones = usize::from(moves_remaining);
    for (k, &(q, r, p)) in future.iter().enumerate() {
        if p != 1 && p != -1 {
            return refuse(format!("has stone ({q}, {r}) with player {p}"));
        }
        let movers_turn = k < mover_stones || ((k - mover_stones) / TURN_STONES) % 2 == 1;
        let owner = if movers_turn {
            current_player
        } else {
            -current_player
        };
        if p != owner {
            return refuse(format!(
                "has stone {k} ({q}, {r}) by {p} where the turn is {owner}'s"
            ));
        }
        if !cells.insert((q, r)) {
            return refuse(format!("places ({q}, {r}) twice"));
        }
    }
    if let Some(&(q, r, _)) = stones.iter().find(|&&(q, r, _)| cells.contains(&(q, r))) {
        return refuse(format!("places ({q}, {r}), a cell the row already holds"));
    }
    let (end_turns, game_turns) = (
        (usize::from(ply_index) + future.len()).div_ceil(2),
        usize::from(game_length),
    );
    if end_turns > game_turns || (future.len() < FUTURE_HORIZON && end_turns != game_turns) {
        return refuse(format!(
            "ends at turn {end_turns} with {} stones but the game lasted {game_turns} turns",
            future.len()
        ));
    }
    Ok(())
}
