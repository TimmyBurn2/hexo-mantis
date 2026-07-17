use super::core::{Board, Cell, Player, HEX_AXES, hex_distance};

impl Board {
    /// Returns **2-plane views** (2 × TOTAL_CELLS = 722 floats each) and the
    /// window centre (cq, cr) for each cluster.
    ///
    /// Plane 0 = current player's stones in this 19×19 window.
    /// Plane 1 = opponent's stones in this 19×19 window.
    ///
    /// # Why 2 planes, not 18?
    ///
    /// The full dense NN input is 18 planes:
    ///   - planes  0-7:  current player's stones at t, t-1, … t-7
    ///   - planes  8-15: opponent's stones        at t, t-1, … t-7
    ///   - planes 16-17: game-state scalars (moves_remaining, ply parity)
    ///
    /// (Chain-length planes live in a separate aux sub-buffer.)
    ///
    /// Assembling all 18 planes requires the full move history, which the
    /// Board does not carry — and shipping 6 498 floats per cluster
    /// (18 × 361) across a boundary when 14 planes are always zero for
    /// engine-driven self-play would be a significant overhead.
    ///
    /// Instead:
    ///   - `get_cluster_views` returns the 2-plane current snapshot (722 floats).
    ///   - history-aware callers stack the current snapshot with historical
    ///     snapshots they own to form the final (18, 19, 19) tensor.
    ///   - the dense 18-plane expansion kernels live in the encoding crate
    ///     and share this snapshot's window semantics.
    // Truncate-toward-zero semantics preserves anchor calibration for legacy
    // checkpoints (predecessor forensic record; falsified-register midpoint
    // row; pinned by inv18b).
    #[allow(clippy::manual_midpoint)]
    pub fn get_cluster_views(&self) -> (Vec<Vec<f32>>, Vec<(i32, i32)>) {
        // Window dimensions resolve from `self.cluster_window_size`
        // (default 19 = v6 wire format; wider callers set 25). The "small
        // cluster" span threshold scales with the window: window − 4 leaves a
        // 2-cell margin around the cluster bbox so the centroid window
        // contains every stone.
        let window_size = self.cluster_window_size;
        let total_cells = window_size * window_size;
        let half: i32 = (window_size as i32 - 1) / 2;
        let span_threshold: i32 = window_size as i32 - 4;

        let clusters = self.get_clusters();
        let mut final_centers = Vec::new();

        if clusters.is_empty() {
            final_centers.push((0, 0));
        } else {
            let threat_anchors = self.get_threat_anchors();
            let action_anchors = &self.action_anchors[..self.action_anchors_count];

            for cluster in clusters {
                let mut min_q = i32::MAX;
                let mut max_q = i32::MIN;
                let mut min_r = i32::MAX;
                let mut max_r = i32::MIN;
                for &(q, r) in &cluster {
                    min_q = min_q.min(q);
                    max_q = max_q.max(q);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }

                let span_q = max_q - min_q;
                let span_r = max_r - min_r;

                if span_q <= span_threshold && span_r <= span_threshold {
                    // Small Clusters: single window centered on geometric middle
                    final_centers.push(((min_q + max_q) / 2, (min_r + max_r) / 2));
                } else {
                    // Massive Clusters: window centered on each Action and Threat anchor in the cluster
                    let mut cluster_anchors = Vec::new();

                    // Action anchors in this cluster
                    for &anchor in action_anchors {
                        if cluster.contains(&anchor) {
                            cluster_anchors.push(anchor);
                        }
                    }

                    // Threat anchors in this cluster
                    for &anchor in &threat_anchors {
                        if cluster.contains(&anchor) {
                            cluster_anchors.push(anchor);
                        }
                    }

                    if cluster_anchors.is_empty() {
                        // Fallback if no anchors found
                        final_centers.push(((min_q + max_q) / 2, (min_r + max_r) / 2));
                    } else {
                        // Deduplicate anchors: radius matches v6 baseline (5)
                        // regardless of cluster_window_size — this is a
                        // dedup tolerance, not a perception radius.
                        let mut deduped: Vec<(i32, i32)> = Vec::new();
                        for &a in &cluster_anchors {
                            if !deduped.iter().any(|&d| hex_distance(a.0, a.1, d.0, d.1) <= 5) {
                                deduped.push(a);
                            }
                        }
                        final_centers.extend(deduped);
                    }
                }
            }
        }

        let mut views = Vec::with_capacity(final_centers.len());
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };

        for &(cq, cr) in &final_centers {
            let mut planes_2 = vec![0.0f32; 2 * total_cells];
            for (&(q, r), &cell) in &self.cells {
                let wq = q - cq + half;
                let wr = r - cr + half;
                if wq >= 0
                    && wq < window_size as i32
                    && wr >= 0
                    && wr < window_size as i32
                {
                    let flat = (wq as usize) * window_size + (wr as usize);
                    if cell == my_cell {
                        planes_2[flat] = 1.0;
                    } else if cell == opp_cell {
                        planes_2[total_cells + flat] = 1.0;
                    }
                }
            }
            views.push(planes_2);
        }
        (views, final_centers)
    }

    /// Returns the (q, r) centers of any open 3-in-a-row or 4-in-a-row formations.
    /// A formation is considered "open" if at least one of its ends is empty.
    ///
    /// Performance: Optimized O(Stones) by skipping redundant scans.
    pub fn get_threat_anchors(&self) -> Vec<(i32, i32)> {
        let mut anchors = Vec::with_capacity(8);
        if self.cells.is_empty() {
            return anchors;
        }

        // Pre-collect stones by player to avoid repeated HashMap scans.
        let mut p1_stones = Vec::with_capacity(self.cells.len());
        let mut p2_stones = Vec::with_capacity(self.cells.len());
        for (&pos, &cell) in &self.cells {
            match cell {
                Cell::P1 => p1_stones.push(pos),
                Cell::P2 => p2_stones.push(pos),
                Cell::Empty => {}
            }
        }

        self.append_threat_anchors_for_player(&p1_stones, Cell::P1, &mut anchors);
        self.append_threat_anchors_for_player(&p2_stones, Cell::P2, &mut anchors);

        anchors
    }

    fn append_threat_anchors_for_player(
        &self,
        stones: &[(i32, i32)],
        player_cell: Cell,
        anchors: &mut Vec<(i32, i32)>,
    ) {
        // Track visited (pos, axis) to avoid redundant scans for the same sequence.
        // There are 3 axes, so we can pack (q, r, axis_idx) into a single key if needed,
        // but for simplicity and to avoid allocations, we just check if the previous
        // cell in the direction is the same color.
        for &(q, r) in stones {
            for (dq, dr) in HEX_AXES {
                // Efficiency: Only start scanning if this stone is the *start* of a sequence
                // in this direction. This ensures each sequence is scanned exactly once.
                if self.get(q - dq, r - dr) == player_cell {
                    continue;
                }

                let mut count = 1;
                while self.get(q + dq * count, r + dr * count) == player_cell {
                    count += 1;
                }

                if count == 3 || count == 4 {
                    // Check if open at either end.
                    let open_start = self.get(q - dq, r - dr) == Cell::Empty;
                    let open_end = self.get(q + dq * count, r + dr * count) == Cell::Empty;

                    if open_start || open_end {
                        let center_idx = count / 2;
                        anchors.push((q + dq * center_idx, r + dr * center_idx));
                    }
                }
            }
        }
    }
}
