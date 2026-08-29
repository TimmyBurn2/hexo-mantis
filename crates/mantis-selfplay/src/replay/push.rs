//! Write path for `ReplayBuffer` — single-position `push_impl` and batched
//! `push_game_impl` / `push_many_impl`. Ported verbatim from the predecessor
//! engine's `replay_buffer/push.rs` with the FFI-binding strip: the borrowed
//! array views are already plain slices on the config structs (the contiguity
//! check moved to WP7), the error type becomes `Result<(), String>`, error
//! strings preserved character-for-character. `n_chain_planes()` (a method
//! old-side) becomes the `n_chain_planes` spec field. The relocated
//! reproject/stress tests live in `tests/replay_sample_aux.rs` (R5).
//!
//! R8: >300 LOC by design — the three strided write kernels (single / per-game /
//! per-row metadata) share the ring-head + weight-bucket discipline and are kept
//! together so the byte-layout of every column stays greppable in one file.

use half::f16;
use std::sync::atomic::Ordering;

use super::push_config::{PushGameConfig, PushManyConfig, PushSingleConfig};
use super::ReplayBuffer;

impl ReplayBuffer {
    /// Store a single `(state, chain_planes, policy, outcome, ownership,
    /// winning_line)` sample.
    pub fn push_impl(&mut self, cfg: PushSingleConfig<'_>) -> Result<(), String> {
        let PushSingleConfig {
            state,
            chain_planes,
            policy,
            outcome,
            ownership,
            winning_line,
            game_id,
            game_length,
            is_full_search,
            position_index,
            value_target_valid,
        } = cfg;

        let state_slice = state;
        let chain_slice = chain_planes;
        let policy_slice = policy;
        let own_slice = ownership;
        let wl_slice = winning_line;

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();
        let n_planes = self.encoding.n_planes;
        let trunk_size = self.encoding.trunk_size;

        if state_slice.len() != state_stride {
            return Err(format!(
                "state must have {} elements ({}×{}×{}, HEXB {}), got {}",
                state_stride,
                n_planes,
                trunk_size,
                trunk_size,
                self.encoding.name,
                state_slice.len()
            ));
        }
        if chain_slice.len() != chain_stride {
            return Err(format!(
                "chain_planes must have {} elements ({}×{}×{}), got {}",
                chain_stride,
                self.encoding.n_chain_planes,
                trunk_size,
                trunk_size,
                chain_slice.len()
            ));
        }
        if policy_slice.len() != policy_stride {
            return Err(format!(
                "policy must have {} elements ({}), got {}",
                policy_stride,
                policy_stride,
                policy_slice.len()
            ));
        }
        if own_slice.len() != aux_stride {
            return Err(format!(
                "ownership must have {} elements ({}), got {}",
                aux_stride,
                aux_stride,
                own_slice.len()
            ));
        }
        if wl_slice.len() != aux_stride {
            return Err(format!(
                "winning_line must have {} elements ({}), got {}",
                aux_stride,
                aux_stride,
                wl_slice.len()
            ));
        }

        let slot = self.head;

        // If overwriting a valid slot, decrement its bucket before we clobber it.
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }

        // Copy state as raw f16 bits.
        let dst = &mut self.states[slot * state_stride..(slot + 1) * state_stride];
        for (d, s) in dst.iter_mut().zip(state_slice.iter()) {
            *d = s.to_bits();
        }

        // Copy chain planes as raw f16 bits.
        let dst_chain = &mut self.chain_planes[slot * chain_stride..(slot + 1) * chain_stride];
        for (d, s) in dst_chain.iter_mut().zip(chain_slice.iter()) {
            *d = s.to_bits();
        }

        self.policies[slot * policy_stride..(slot + 1) * policy_stride]
            .copy_from_slice(policy_slice);
        self.outcomes[slot] = outcome;
        self.game_ids[slot] = game_id;
        self.weights[slot] = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };

        self.ownership[slot * aux_stride..(slot + 1) * aux_stride].copy_from_slice(own_slice);
        self.winning_line[slot * aux_stride..(slot + 1) * aux_stride].copy_from_slice(wl_slice);
        self.is_full_search[slot] = is_full_search as u8;
        self.value_target_valid[slot] = value_target_valid as u8;
        self.position_indices[slot] = position_index;

        // Increment the new position's bucket.
        let new_bucket = Self::weight_bucket(self.weights[slot]);
        self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);

        // R245(c): derive this row's losslessness flag from the row just written.
        self.refresh_compact(slot);

        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
        Ok(())
    }

    /// Store all positions from a completed game efficiently (handles wrap).
    pub fn push_game_impl(&mut self, cfg: PushGameConfig<'_>) -> Result<(), String> {
        let PushGameConfig {
            states,
            chain_planes,
            policies,
            outcomes,
            ownership,
            winning_line,
            game_id,
            game_length,
            is_full_search,
            position_indices,
            value_target_valid,
        } = cfg;

        let states_s = states;
        let chain_s = chain_planes;
        let policies_s = policies;
        let outcomes_s = outcomes;
        let own_s = ownership;
        let wl_s = winning_line;
        // Resolve optional is_full_search slice; default 1 (full-search) per row.
        let ifs_s: &[u8] = is_full_search.unwrap_or(&[]);
        // Resolve optional position_indices slice; None ⇒ fills 0.
        let pos_s: &[u16] = position_indices.unwrap_or(&[]);
        // Resolve optional value_target_valid slice; default 1 (supervise).
        let vv_s: &[u8] = value_target_valid.unwrap_or(&[]);

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        let t = outcomes_s.len();
        if t == 0 {
            return Ok(());
        }

        if states_s.len() != t * state_stride {
            return Err("states shape mismatch".to_string());
        }
        if chain_s.len() != t * chain_stride {
            return Err("chain_planes shape mismatch".to_string());
        }
        if policies_s.len() != t * policy_stride {
            return Err("policies shape mismatch".to_string());
        }
        if own_s.len() != t * aux_stride {
            return Err("ownership shape mismatch".to_string());
        }
        if wl_s.len() != t * aux_stride {
            return Err("winning_line shape mismatch".to_string());
        }
        if !ifs_s.is_empty() && ifs_s.len() != t {
            return Err(format!(
                "is_full_search must have {} elements (one per position), got {}",
                t,
                ifs_s.len()
            ));
        }
        if !pos_s.is_empty() && pos_s.len() != t {
            return Err(format!(
                "position_indices must have {} elements (one per position), got {}",
                t,
                pos_s.len()
            ));
        }
        if !vv_s.is_empty() && vv_s.len() != t {
            return Err(format!(
                "value_target_valid must have {} elements (one per position), got {}",
                t,
                vv_s.len()
            ));
        }

        let w = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };
        let new_bucket = Self::weight_bucket(w);

        let available = self.capacity.saturating_sub(self.size);

        for i in 0..t {
            let slot = (self.head + i) % self.capacity;

            if i >= available {
                let old_bucket = Self::weight_bucket(self.weights[slot]);
                self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
            }

            // State: convert f16 → u16 bits
            let src_state = &states_s[i * state_stride..(i + 1) * state_stride];
            let dst_state = &mut self.states[slot * state_stride..(slot + 1) * state_stride];
            for (d, s) in dst_state.iter_mut().zip(src_state.iter()) {
                *d = s.to_bits();
            }

            // Chain planes: convert f16 → u16 bits
            let src_chain = &chain_s[i * chain_stride..(i + 1) * chain_stride];
            let dst_chain = &mut self.chain_planes[slot * chain_stride..(slot + 1) * chain_stride];
            for (d, s) in dst_chain.iter_mut().zip(src_chain.iter()) {
                *d = s.to_bits();
            }

            // Policy: direct f32 copy
            let src_pol = &policies_s[i * policy_stride..(i + 1) * policy_stride];
            self.policies[slot * policy_stride..(slot + 1) * policy_stride]
                .copy_from_slice(src_pol);

            // Auxiliary spatial targets — direct u8 copies.
            self.ownership[slot * aux_stride..(slot + 1) * aux_stride]
                .copy_from_slice(&own_s[i * aux_stride..(i + 1) * aux_stride]);
            self.winning_line[slot * aux_stride..(slot + 1) * aux_stride]
                .copy_from_slice(&wl_s[i * aux_stride..(i + 1) * aux_stride]);

            // is_full_search: use provided value or default to 1 (full-search).
            self.is_full_search[slot] = if ifs_s.is_empty() { 1u8 } else { ifs_s[i] };
            // position_index: use provided value or default 0 (matches push_many).
            self.position_indices[slot] = if pos_s.is_empty() { 0u16 } else { pos_s[i] };
            // value_target_valid: provided value or default 1 (supervise).
            self.value_target_valid[slot] = if vv_s.is_empty() { 1u8 } else { vv_s[i] };

            self.outcomes[slot] = outcomes_s[i];
            self.game_ids[slot] = game_id;
            self.weights[slot] = w;

            self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);

            // R245(c): per-row losslessness flag, derived from the row just written.
            self.refresh_compact(slot);
        }

        self.head = (self.head + t) % self.capacity;
        self.size = (self.size + t).min(self.capacity);
        Ok(())
    }

    /// Store N positions with per-row `game_length` and `is_full_search`.
    pub fn push_many_impl(&mut self, cfg: PushManyConfig<'_>) -> Result<(), String> {
        let PushManyConfig {
            states,
            chain_planes,
            policies,
            outcomes,
            ownership,
            winning_line,
            game_lengths,
            is_full_search,
            position_indices,
            value_target_valid,
        } = cfg;

        let states_s = states;
        let chain_s = chain_planes;
        let policies_s = policies;
        let outcomes_s = outcomes;
        let own_s = ownership;
        let wl_s = winning_line;
        let gl_s = game_lengths;
        let ifs_s = is_full_search;
        let pos_s: &[u16] = position_indices.unwrap_or(&[]);
        let vv_s: &[u8] = value_target_valid.unwrap_or(&[]);

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        let t = outcomes_s.len();
        if t == 0 {
            return Ok(());
        }

        if states_s.len() != t * state_stride {
            return Err("states shape mismatch".to_string());
        }
        if chain_s.len() != t * chain_stride {
            return Err("chain_planes shape mismatch".to_string());
        }
        if policies_s.len() != t * policy_stride {
            return Err("policies shape mismatch".to_string());
        }
        if own_s.len() != t * aux_stride {
            return Err("ownership shape mismatch".to_string());
        }
        if wl_s.len() != t * aux_stride {
            return Err("winning_line shape mismatch".to_string());
        }
        if gl_s.len() != t {
            return Err(format!(
                "game_lengths must have {t} elements, got {}",
                gl_s.len()
            ));
        }
        if ifs_s.len() != t {
            return Err(format!(
                "is_full_search must have {t} elements, got {}",
                ifs_s.len()
            ));
        }
        if !pos_s.is_empty() && pos_s.len() != t {
            return Err(format!(
                "position_indices must have {t} elements, got {}",
                pos_s.len()
            ));
        }
        if !vv_s.is_empty() && vv_s.len() != t {
            return Err(format!(
                "value_target_valid must have {t} elements, got {}",
                vv_s.len()
            ));
        }

        let available = self.capacity.saturating_sub(self.size);

        // f16 and u16 are same size/align; to_bits() is a native-endian transmute.
        debug_assert_eq!(std::mem::size_of::<f16>(), std::mem::size_of::<u16>());
        debug_assert_eq!(std::mem::align_of::<f16>(), std::mem::align_of::<u16>());

        for i in 0..t {
            let slot = (self.head + i) % self.capacity;

            if i >= available {
                let old_bucket = Self::weight_bucket(self.weights[slot]);
                self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
            }

            let game_length = gl_s[i];
            let w = if game_length == 0 {
                f16::from_f32(1.0).to_bits()
            } else {
                self.weight_schedule.weight_for(game_length)
            };
            let new_bucket = Self::weight_bucket(w);

            let src_state = &states_s[i * state_stride..(i + 1) * state_stride];
            let dst_state = &mut self.states[slot * state_stride..(slot + 1) * state_stride];
            debug_assert_eq!(src_state.len(), dst_state.len());
            // SAFETY: f16 and u16 same size/align; bit pattern preserved (to_bits semantics).
            let src_bits = unsafe {
                std::slice::from_raw_parts(src_state.as_ptr().cast::<u16>(), src_state.len())
            };
            dst_state.copy_from_slice(src_bits);

            let src_chain = &chain_s[i * chain_stride..(i + 1) * chain_stride];
            let dst_chain = &mut self.chain_planes[slot * chain_stride..(slot + 1) * chain_stride];
            debug_assert_eq!(src_chain.len(), dst_chain.len());
            // SAFETY: same as above.
            let chain_bits = unsafe {
                std::slice::from_raw_parts(src_chain.as_ptr().cast::<u16>(), src_chain.len())
            };
            dst_chain.copy_from_slice(chain_bits);

            self.policies[slot * policy_stride..(slot + 1) * policy_stride]
                .copy_from_slice(&policies_s[i * policy_stride..(i + 1) * policy_stride]);
            self.ownership[slot * aux_stride..(slot + 1) * aux_stride]
                .copy_from_slice(&own_s[i * aux_stride..(i + 1) * aux_stride]);
            self.winning_line[slot * aux_stride..(slot + 1) * aux_stride]
                .copy_from_slice(&wl_s[i * aux_stride..(i + 1) * aux_stride]);

            self.is_full_search[slot] = ifs_s[i];
            self.position_indices[slot] = if pos_s.is_empty() { 0u16 } else { pos_s[i] };
            self.value_target_valid[slot] = if vv_s.is_empty() { 1u8 } else { vv_s[i] };
            self.outcomes[slot] = outcomes_s[i];
            self.game_ids[slot] = -1;
            self.weights[slot] = w;

            self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);

            // R245(c): per-row losslessness flag, derived from the row just written.
            self.refresh_compact(slot);
        }

        self.head = (self.head + t) % self.capacity;
        self.size = (self.size + t).min(self.capacity);
        Ok(())
    }
}
