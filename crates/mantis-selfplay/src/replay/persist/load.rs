//! R8: >300 LOC by design — the four-version header dispatch + guards + the
//! per-row payload reader are one load path kept together verbatim for byte-exact
//! version-compat audit.
//!
//! HEXB v9 + v8 + v7 + v6 load path for `ReplayBuffer::load_from_path`. Ported
//! verbatim from the predecessor engine's `replay_buffer/persist/load.rs`; it
//! already returned `Result<usize, String>` (no bindings to strip). v9 added
//! `value_target_valid: u8` per-row; v8 added `position_index: u16`; older files
//! load with those columns defaulted (1 / 0). v5 and earlier are hard-rejected.

use std::sync::atomic::Ordering;

use super::{ReplayBuffer, HEXB_MAGIC, HEXB_VERSION};

impl ReplayBuffer {
    /// Load buffer contents from a binary file written by `save_to_path`.
    ///
    /// Returns the number of positions loaded. If the file does not exist,
    /// returns 0 (not an error — supports first-run). If the saved buffer has
    /// more positions than `self.capacity`, only the most recent `self.capacity`
    /// are loaded.
    #[allow(clippy::too_many_lines)]
    pub fn load_from_path(&mut self, path: &str) -> Result<usize, String> {
        use std::io::{BufReader, Read};

        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(format!("cannot open {path}: {e}")),
        };
        let mut r = BufReader::new(file);

        // Read header
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4).map_err(|e| format!("{e}"))?;
        let magic = u32::from_le_bytes(buf4);
        if magic != HEXB_MAGIC {
            return Err(format!("invalid magic: expected 0x48455842 (HEXB), got 0x{magic:08X}"));
        }

        r.read_exact(&mut buf4).map_err(|e| format!("{e}"))?;
        let version = u32::from_le_bytes(buf4);

        // ── Version dispatch ──────────────────────────────────────────────
        let file_encoding_name: String;
        let saved_n_planes: usize;
        let _saved_capacity: usize;
        let saved_size: usize;

        if (7..=9).contains(&version) {
            // v7/v8/v9 header: n_planes, capacity, size, encoding_name_len, encoding_name.
            r.read_exact(&mut buf4).map_err(|e| format!("{e}"))?;
            saved_n_planes = u32::from_le_bytes(buf4) as usize;

            r.read_exact(&mut buf8).map_err(|e| format!("{e}"))?;
            _saved_capacity = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf8).map_err(|e| format!("{e}"))?;
            saved_size = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf4).map_err(|e| format!("{e}"))?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            if name_len > 256 {
                return Err(format!("HEXB v{version} encoding_name_len={name_len} exceeds maximum 256"));
            }
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf).map_err(|e| format!("{e}"))?;
            file_encoding_name = String::from_utf8(name_buf)
                .map_err(|e| format!("HEXB v{version} encoding name is not valid UTF-8: {e}"))?;
        } else if version == 6 {
            // v6 backward compat: no encoding field — assume "v6"
            eprintln!(
                "warning: loading deprecated HEXB v6 file ({path}). \
                 Assuming encoding 'v6'. Re-save to upgrade to v7."
            );
            r.read_exact(&mut buf4).map_err(|e| format!("{e}"))?;
            saved_n_planes = u32::from_le_bytes(buf4) as usize;

            r.read_exact(&mut buf8).map_err(|e| format!("{e}"))?;
            _saved_capacity = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf8).map_err(|e| format!("{e}"))?;
            saved_size = u64::from_le_bytes(buf8) as usize;

            // silent-encoding-gate: ok -- not a fallback. This arm is reached only when the
            // file's own header declares HEXB wire version 6, and the `else` below returns
            // an error; v6 is what that version IS, derived from the file rather than
            // substituted for an absent value. (WPUF-2 R58 / REVIEW_IMPL_R2 A3.)
            file_encoding_name = "v6".to_string();
        } else {
            return Err(format!(
                "HEXB version {version} not supported (this build only reads v{HEXB_VERSION}). \
                 v5 and earlier are deprecated by the §122 B4 verdict — wire-format \
                 channel-drop (18 → 8 planes per D17 ablation Set A) is incompatible \
                 with their plane count. Regenerate the buffer with v{HEXB_VERSION}."
            ));
        }

        // ── Encoding validation ───────────────────────────────────────────
        let Some(file_spec) = mantis_encoding::registry::lookup(&file_encoding_name) else {
            return Err(format!(
                "HEXB file declares unknown encoding '{file_encoding_name}'. \
                 Registered encodings: {:?}",
                {
                    let mut known: Vec<&str> =
                        mantis_encoding::registry::all_specs().map(|s| s.name).collect();
                    known.sort_unstable();
                    known
                }
            ));
        };

        if saved_n_planes != file_spec.n_planes {
            return Err(format!(
                "HEXB file header has n_planes={saved_n_planes}, \
                 but encoding '{file_encoding_name}' expects {}. \
                 File may be corrupted or written with a mismatched registry.",
                file_spec.n_planes
            ));
        }

        // Cross-encoding mismatch guard (§P13).
        //
        // Compare wire-format signature instead of the encoding name string.
        // Two encodings producing byte-identical HEXB rows must auto-cross-load
        // even when their names differ. Any signature drift (different n_planes,
        // board_size, policy_logit_count, has_pass_slot, or sym_table_id) still
        // hard-errors.
        let buffer_sig = self.encoding.wire_signature();
        let file_sig = file_spec.wire_signature();
        if buffer_sig != file_sig {
            return Err(format!(
                "HEXB encoding mismatch: buffer encoding '{}' wire_signature \
                 {:?} differs from file encoding '{}' wire_signature {:?}",
                self.encoding.name, buffer_sig, file_encoding_name, file_sig
            ));
        }

        // How many to actually load — cap at our capacity
        let to_load = saved_size.min(self.capacity);
        // How many to skip if saved_size > capacity (skip oldest)
        let to_skip = saved_size - to_load;

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        // Per-entry byte sizes.
        let state_bytes = state_stride * 2;
        let chain_bytes = chain_stride * 2;
        let policy_bytes = policy_stride * 4;
        // v6/v7 entry: state + chain + policy + outcome(4) + game_id(8) + weight(2)
        //              + ownership + winning_line + is_full_search(1)
        // v8 entry: above + position_index(2); v9 entry: above + value_target_valid(1)
        let v7_entry_bytes = state_bytes + chain_bytes + policy_bytes + 4 + 8 + 2 + aux_stride + aux_stride + 1;
        let entry_bytes = match version {
            9 => v7_entry_bytes + 2 + 1,
            8 => v7_entry_bytes + 2,
            _ => v7_entry_bytes,
        };

        // Skip oldest entries
        if to_skip > 0 {
            let skip_bytes = to_skip * entry_bytes;
            let mut remaining = skip_bytes;
            let mut skip_buf = vec![0u8; 8192.min(skip_bytes)];
            while remaining > 0 {
                let chunk = remaining.min(skip_buf.len());
                r.read_exact(&mut skip_buf[..chunk]).map_err(|e| format!("{e}"))?;
                remaining -= chunk;
            }
        }

        // Reset weight histogram
        for bucket in &self.weight_buckets {
            bucket.store(0, Ordering::Relaxed);
        }

        // Read positions directly into storage
        let mut state_buf = vec![0u8; state_bytes];
        let mut chain_buf = vec![0u8; chain_bytes];
        let mut pol_buf = vec![0u8; policy_bytes];

        for i in 0..to_load {
            let slot = i; // write sequentially from slot 0

            // state
            r.read_exact(&mut state_buf).map_err(|e| format!("{e}"))?;
            let dst_state = &mut self.states[slot * state_stride..(slot + 1) * state_stride];
            for (j, d) in dst_state.iter_mut().enumerate() {
                *d = u16::from_le_bytes([state_buf[j * 2], state_buf[j * 2 + 1]]);
            }

            // chain_planes
            r.read_exact(&mut chain_buf).map_err(|e| format!("{e}"))?;
            let dst_chain = &mut self.chain_planes[slot * chain_stride..(slot + 1) * chain_stride];
            for (j, d) in dst_chain.iter_mut().enumerate() {
                *d = u16::from_le_bytes([chain_buf[j * 2], chain_buf[j * 2 + 1]]);
            }

            // policy
            r.read_exact(&mut pol_buf).map_err(|e| format!("{e}"))?;
            let dst_pol = &mut self.policies[slot * policy_stride..(slot + 1) * policy_stride];
            for (j, d) in dst_pol.iter_mut().enumerate() {
                *d = f32::from_le_bytes([
                    pol_buf[j * 4],
                    pol_buf[j * 4 + 1],
                    pol_buf[j * 4 + 2],
                    pol_buf[j * 4 + 3],
                ]);
            }

            // outcome
            r.read_exact(&mut buf4).map_err(|e| format!("{e}"))?;
            self.outcomes[slot] = f32::from_le_bytes(buf4);

            // game_id
            r.read_exact(&mut buf8).map_err(|e| format!("{e}"))?;
            self.game_ids[slot] = i64::from_le_bytes(buf8);

            // weight
            let mut buf2 = [0u8; 2];
            r.read_exact(&mut buf2).map_err(|e| format!("{e}"))?;
            let w_bits = u16::from_le_bytes(buf2);
            self.weights[slot] = w_bits;

            // ownership + winning_line
            let aux_dst_start = slot * aux_stride;
            r.read_exact(&mut self.ownership[aux_dst_start..aux_dst_start + aux_stride])
                .map_err(|e| format!("{e}"))?;
            r.read_exact(&mut self.winning_line[aux_dst_start..aux_dst_start + aux_stride])
                .map_err(|e| format!("{e}"))?;

            // is_full_search: u8
            let mut buf1 = [0u8; 1];
            r.read_exact(&mut buf1).map_err(|e| format!("{e}"))?;
            self.is_full_search[slot] = buf1[0];

            // position_index per row (v8+ only). v6/v7 default to 0.
            if version >= 8 {
                let mut buf2 = [0u8; 2];
                r.read_exact(&mut buf2).map_err(|e| format!("{e}"))?;
                self.position_indices[slot] = u16::from_le_bytes(buf2);
            } else {
                self.position_indices[slot] = 0;
            }

            // value_target_valid per row (v9+ only). Older versions default to 1
            // explicitly (the destination buffer may be reused, so relying on
            // new()'s all-ones init is not enough).
            if version == 9 {
                r.read_exact(&mut buf1).map_err(|e| format!("{e}"))?;
                self.value_target_valid[slot] = buf1[0];
            } else {
                self.value_target_valid[slot] = 1;
            }

            // Update weight histogram
            let bucket = Self::weight_bucket(w_bits);
            self.weight_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }

        // R245(c): the losslessness flag is NOT on the wire (no on-disk format
        // change — a HEXB file written by any prior build still loads) and is
        // RECOMPUTED here for every row that landed, after the whole row is in
        // place. A slot the loaded file did not reach keeps whatever flag it held;
        // it is outside `size` and unreachable by `sample_indices`.
        for slot in 0..to_load {
            self.refresh_compact(slot);
        }

        self.size = to_load;
        self.head = to_load % self.capacity;
        Ok(to_load)
    }
}

#[cfg(test)]
mod tests {
    use super::ReplayBuffer;
    use mantis_encoding::registry::lookup_or_panic;
    use mantis_encoding::RegistrySpec;

    fn unique_test_path(stem: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hexb_o9_{stem}_{pid}_{nanos}_{n}.hexb"))
    }

    /// O-9 — P13 accept-on-name-mismatch witness (C-7). A file declaring the
    /// registered name "v6" LOADS into a buffer whose `encoding` is a leaked Copy
    /// of `lookup("v6")` with `.name = "v6_alias"` (wire_signature identical to
    /// v6, name divergent). The load guard compares `wire_signature()`, NOT the
    /// name string — this MUST FAIL if a future edit made the guard also compare
    /// names. Requires the `#[cfg(test)]`-only `with_encoding` constructor (never
    /// a production surface); the alias is a synthetic leaked local, never
    /// registered (NOT a v7full/v8 KILL resurrection).
    #[test]
    fn o9_accept_on_name_mismatch_when_wire_signature_matches() {
        let v6 = lookup_or_panic("v6");
        // name-divergent, wire-signature-identical to v6:
        let mut aliased = *v6; // RegistrySpec is Copy
        aliased.name = "v6_alias";
        let aliased: &'static RegistrySpec = Box::leak(Box::new(aliased));
        assert_ne!(aliased.name, "v6");
        assert_eq!(
            aliased.wire_signature(),
            v6.wire_signature(),
            "alias must keep v6's wire_signature (only the name diverges)"
        );
        // The alias is not registered — the guard keys on the FILE's name, not this.
        assert!(mantis_encoding::registry::lookup("v6_alias").is_none());

        let mut writer = ReplayBuffer::new(8, "v6"); // file DECLARES the registered "v6"
        writer.push_for_test(1.0, 10, true);
        let path = unique_test_path("v6_alias");
        writer.save_to_path(path.to_str().unwrap()).unwrap();

        let mut reader = ReplayBuffer::with_encoding(8, aliased); // BUFFER name = "v6_alias"
        let n = reader
            .load_from_path(path.to_str().unwrap())
            .expect("name-divergent buffer with matching wire_signature must LOAD");
        assert_eq!(n, 1, "the v6 file must load into the v6_alias buffer (sig == v6.sig)");

        let _ = std::fs::remove_file(path);
    }
}
