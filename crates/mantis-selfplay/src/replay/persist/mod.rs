//! HEXB v9 on-disk format for `ReplayBuffer` — `save_to_path` and
//! `load_from_path`. Ported from the predecessor engine's `replay_buffer/persist/`
//! with the FFI-binding strip (the error type becomes `Result<_, String>`; error
//! strings preserved verbatim). The write-side `&[u16] as &[u8]` / `&[f32] as
//! &[u8]` native-endian dump stays (binding-free; endianness is a pre-existing property —
//! save is native-endian, load is explicit little-endian, so the format is
//! correct only on little-endian hosts, ported verbatim).
//!
//! Format (little-endian native):
//!   [magic: u32 = 0x48455842]  ("HEXB")
//!   [version: u32 = 9]
//!   [n_planes: u32]            (redundant sanity field, must equal encoding.n_planes)
//!   [capacity: u64]
//!   [size: u64]
//!   [encoding_name_len: u32]   (added in v7)
//!   [encoding_name: [u8; N]]   (UTF-8, no null terminator)
//!   For each of `size` positions (oldest → newest):
//!     state:          state_stride × u16    (n_planes × n_cells)
//!     chain_planes:   chain_stride × u16    (n_chain_planes × n_cells)
//!     policy:         policy_stride × f32
//!     outcome:        f32
//!     game_id:        i64
//!     weight:         u16
//!     ownership:      aux_stride × u8
//!     winning_line:   aux_stride × u8
//!     is_full_search: u8
//!     position_index: u16                    (added in v8)
//!     value_target_valid: u8                 (added in v9)
//!
//! Backward compat on load (see `load.rs`): v8 files default value_target_valid=1;
//! v7 files default position_index=0; v6 files lack the encoding_name (assumed
//! "v6" with a deprecation warning); v5 and earlier are HARD-REJECTED.

use super::ReplayBuffer;

mod load;

pub(crate) const HEXB_MAGIC: u32 = 0x4845_5842; // "HEXB"
pub(crate) const HEXB_VERSION: u32 = 9;

impl ReplayBuffer {
    /// Save buffer contents to a binary file in HEXB v9 format.
    pub fn save_to_path(&self, path: &str) -> Result<(), String> {
        use std::io::{BufWriter, Write};

        let file = std::fs::File::create(path).map_err(|e| format!("cannot create {path}: {e}"))?;
        let mut w = BufWriter::new(file);

        // Header
        w.write_all(&HEXB_MAGIC.to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.write_all(&HEXB_VERSION.to_le_bytes())
            .map_err(|e| e.to_string())?;
        // Redundant plane-count field for sanity checking.
        w.write_all(&(self.encoding.n_planes as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.write_all(&(self.capacity as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.write_all(&(self.size as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;

        // v7: encoding name
        let name_bytes = self.encoding.name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.write_all(name_bytes).map_err(|e| e.to_string())?;

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        // Positions in logical order (oldest → newest)
        for i in 0..self.size {
            let slot = (self.head + self.capacity - self.size + i) % self.capacity;

            // state: u16 slice → bytes
            let state_start = slot * state_stride;
            // SAFETY: &[u16] is layout-compatible with &[u8] (byte_len = 2 × elem_len);
            // resulting slice lifetime is bounded here and consumed by w.write_all
            // before any aliasing &mut to self.states can be created.
            let state_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.states[state_start..state_start + state_stride]
                        .as_ptr()
                        .cast::<u8>(),
                    state_stride * 2,
                )
            };
            w.write_all(state_bytes).map_err(|e| e.to_string())?;

            // chain_planes: u16 slice → bytes
            let chain_start = slot * chain_stride;
            // SAFETY: as above.
            let chain_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.chain_planes[chain_start..chain_start + chain_stride]
                        .as_ptr()
                        .cast::<u8>(),
                    chain_stride * 2,
                )
            };
            w.write_all(chain_bytes).map_err(|e| e.to_string())?;

            // policy: f32 slice → bytes
            let pol_start = slot * policy_stride;
            // SAFETY: &[f32] is layout-compatible with &[u8] (byte_len = 4 × elem_len).
            let pol_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.policies[pol_start..pol_start + policy_stride]
                        .as_ptr()
                        .cast::<u8>(),
                    policy_stride * 4,
                )
            };
            w.write_all(pol_bytes).map_err(|e| e.to_string())?;

            // outcome: f32
            w.write_all(&self.outcomes[slot].to_le_bytes())
                .map_err(|e| e.to_string())?;
            // game_id: i64
            w.write_all(&self.game_ids[slot].to_le_bytes())
                .map_err(|e| e.to_string())?;
            // weight: u16
            w.write_all(&self.weights[slot].to_le_bytes())
                .map_err(|e| e.to_string())?;

            // ownership: aux_stride × u8
            let aux_start = slot * aux_stride;
            w.write_all(&self.ownership[aux_start..aux_start + aux_stride])
                .map_err(|e| e.to_string())?;
            // winning_line: aux_stride × u8
            w.write_all(&self.winning_line[aux_start..aux_start + aux_stride])
                .map_err(|e| e.to_string())?;
            // is_full_search: u8
            w.write_all(&[self.is_full_search[slot]])
                .map_err(|e| e.to_string())?;
            // position_index u16 (HEXB v8)
            w.write_all(&self.position_indices[slot].to_le_bytes())
                .map_err(|e| e.to_string())?;
            // value_target_valid u8 (HEXB v9)
            w.write_all(&[self.value_target_valid[slot]])
                .map_err(|e| e.to_string())?;
        }

        w.flush().map_err(|e| e.to_string())?;
        Ok(())
    }
}
