// >300 justify (R8): the served forward is one pass, its loader and its batch check — split, a module would hold
// half of a shape contract.

//! `RustForward`: the export's served GINE forward over contract #13's batch, f32, single thread.

use std::fmt;

use crate::batch::WebBatch;

/// One row of the manifest's tensor table: `rows × cols` floats at `offset` in the weights file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRow {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub offset: usize,
}

/// The net's outputs for one batch: `logits` and `probs` over its legal rows, `value` over its graphs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NetOutputs {
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub value: Vec<f32>,
}

/// Why the forward refused its weights or a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardError {
    Missing(String),
    Shape {
        name: String,
        want: String,
        got: (usize, usize),
    },
    OutOfFile {
        name: String,
        end: usize,
        len: usize,
    },
    Batch(String),
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(name) => write!(f, "rust forward: the tensor table has no `{name}`"),
            Self::Shape { name, want, got } => {
                write!(f, "rust forward: `{name}` is {got:?}, wants {want}")
            }
            Self::OutOfFile { name, end, len } => {
                write!(
                    f,
                    "rust forward: `{name}` ends at float {end} of a {len}-float weights file"
                )
            }
            Self::Batch(why) => write!(f, "rust forward: {why}"),
        }
    }
}

impl std::error::Error for ForwardError {}

struct Linear {
    wt: Vec<f32>,
    b: Vec<f32>,
    k: usize,
    n: usize,
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    norm: Norm,
    table: Vec<f32>,
    one_eps: f32,
    mlp0: Linear,
    mlp2: Linear,
}

/// The served forward's weights, shape-checked against each other at load.
pub struct RustForward {
    feat: usize,
    hidden: usize,
    codes: usize,
    input_proj: Linear,
    layers: Vec<Layer>,
    final_norm: Norm,
    policy0: Linear,
    policy2: Linear,
    fc1: Linear,
    fc2: Linear,
    support: Vec<f32>,
}

struct Weights<'a> {
    table: &'a [TensorRow],
    floats: &'a [f32],
}

impl Weights<'_> {
    fn get(
        &self,
        name: &str,
        rows: Option<usize>,
        cols: Option<usize>,
    ) -> Result<(usize, usize, Vec<f32>), ForwardError> {
        let row = self
            .table
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| ForwardError::Missing(name.to_owned()))?;
        if rows.is_some_and(|r| r != row.rows) || cols.is_some_and(|c| c != row.cols) {
            return Err(ForwardError::Shape {
                name: name.to_owned(),
                want: format!("{rows:?} × {cols:?}"),
                got: (row.rows, row.cols),
            });
        }
        let end = row
            .rows
            .checked_mul(row.cols)
            .and_then(|n| n.checked_add(row.offset))
            .unwrap_or(usize::MAX);
        let data = self
            .floats
            .get(row.offset..end)
            .ok_or(ForwardError::OutOfFile {
                name: name.to_owned(),
                end,
                len: self.floats.len(),
            })?;
        Ok((row.rows, row.cols, data.to_vec()))
    }

    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, ForwardError> {
        Ok(self.get(name, Some(1), Some(len))?.2)
    }

    fn linear(
        &self,
        name: &str,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Result<Linear, ForwardError> {
        let (k, n, wt) = self.get(&format!("{name}.wt"), k, n)?;
        Ok(Linear {
            wt,
            b: self.vector(&format!("{name}.b"), n)?,
            k,
            n,
        })
    }

    fn norm(&self, name: &str, h: usize) -> Result<Norm, ForwardError> {
        Ok(Norm {
            w: self.vector(&format!("{name}.w"), h)?,
            b: self.vector(&format!("{name}.b"), h)?,
        })
    }
}

fn count_of(n: i32, what: &str) -> Result<usize, ForwardError> {
    usize::try_from(n).map_err(|_| ForwardError::Batch(format!("a negative {what} ({n})")))
}

/// Each array's indices in range and its lengths consistent, so the pass below indexes without a check.
fn check(batch: &WebBatch, feat: usize, codes: usize) -> Result<usize, ForwardError> {
    let bad = |why: String| Err(ForwardError::Batch(why));
    if feat == 0 || !batch.x.len().is_multiple_of(feat) {
        return bad(format!(
            "x holds {} floats, not rows of {feat}",
            batch.x.len()
        ));
    }
    let (n, g, e, l) = (
        batch.x.len() / feat,
        batch.n_graphs,
        batch.src.len(),
        batch.legal.len(),
    );
    for (what, len, want) in [
        ("dst", batch.dst.len(), e),
        ("code", batch.code.len(), e),
        ("node_graph", batch.node_graph.len(), n),
        ("divisor", batch.divisor.len(), n),
        ("real", batch.real.len(), n),
        ("mean_mask", batch.mean_mask.len(), n),
        ("legal_graph", batch.legal_graph.len(), l),
        ("mean_den", batch.mean_den.len(), g),
    ] {
        if len != want {
            return bad(format!("{what} holds {len}, wants {want}"));
        }
    }
    for (what, values, bound) in [
        ("src", &batch.src, n),
        ("dst", &batch.dst, n),
        ("legal", &batch.legal, n),
        ("code", &batch.code, codes),
        ("node_graph", &batch.node_graph, g),
        ("legal_graph", &batch.legal_graph, g),
    ] {
        for &v in values {
            if count_of(v, what)? >= bound {
                return bad(format!("{what} {v} is past {bound}"));
            }
        }
    }
    if batch.divisor.iter().chain(&batch.mean_den).any(|&d| d < 1) {
        return bad("a divisor or a mean count below 1".to_owned());
    }
    Ok(n)
}

impl RustForward {
    /// The weights file read through the manifest's tensor table. Errors: `Missing`, `Shape`, `OutOfFile`.
    pub fn load(
        table: &[TensorRow],
        floats: &[f32],
        feat: usize,
    ) -> Result<RustForward, ForwardError> {
        let w = Weights { table, floats };
        let input_proj = w.linear("input_proj", Some(feat), None)?;
        let hidden = input_proj.n;
        let zero = |name: &str, got: (usize, usize)| ForwardError::Shape {
            name: name.to_owned(),
            want: "a width above 0".to_owned(),
            got,
        };
        if hidden == 0 {
            return Err(zero("input_proj.wt", (input_proj.k, 0)));
        }
        let depth = (0..table.len())
            .take_while(|i| table.iter().any(|r| r.name == format!("table{i}")))
            .count();
        let codes = w.get("table0", None, Some(hidden))?.0;
        let layers = (0..depth)
            .map(|i| {
                Ok(Layer {
                    norm: w.norm(&format!("norm{i}"), hidden)?,
                    table: w.get(&format!("table{i}"), Some(codes), Some(hidden))?.2,
                    one_eps: 1.0 + w.vector(&format!("eps{i}"), 1)?[0],
                    mlp0: w.linear(&format!("mlp{i}.0"), Some(hidden), None)?,
                    mlp2: w.linear(&format!("mlp{i}.2"), None, Some(hidden))?,
                })
            })
            .collect::<Result<Vec<_>, ForwardError>>()?;
        for (i, layer) in layers.iter().enumerate() {
            if layer.mlp0.n != layer.mlp2.k {
                return Err(ForwardError::Shape {
                    name: format!("mlp{i}.2.wt"),
                    want: format!("{} rows", layer.mlp0.n),
                    got: (layer.mlp2.k, layer.mlp2.n),
                });
            }
        }
        let policy0 = w.linear("policy.0", Some(depth * hidden), None)?;
        let policy2 = w.linear("policy.2", Some(policy0.n), Some(1))?;
        let fc1 = w.linear("fc1", Some(2 * depth * hidden), None)?;
        let fc2 = w.linear("fc2", Some(fc1.n), None)?;
        if fc2.n == 0 {
            return Err(zero("fc2.wt", (fc2.k, 0)));
        }
        let support = w.vector("support", fc2.n)?;
        Ok(RustForward {
            feat,
            hidden,
            codes,
            input_proj,
            layers,
            final_norm: w.norm("final_norm", hidden)?,
            policy0,
            policy2,
            fc1,
            fc2,
            support,
        })
    }

    /// The served outputs for `batch`, as `WebServedGraph` computes them. Errors: `Batch` on a malformed array.
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)] // indices are range-checked by `check`
    pub fn forward(&self, batch: &WebBatch) -> Result<NetOutputs, ForwardError> {
        let n = check(batch, self.feat, self.codes)?;
        let (h, g, depth) = (self.hidden, batch.n_graphs, self.layers.len());
        let wide = depth * h;
        let mut x = linear(&batch.x, n, &self.input_proj, false);
        let mut emb = vec![0f32; n * wide];
        let mut agg = vec![0f32; n * h];
        for (i, layer) in self.layers.iter().enumerate() {
            let hn = layer_norm(&x, h, &layer.norm);
            agg.fill(0.0);
            for ((&s, &d), &c) in batch.src.iter().zip(&batch.dst).zip(&batch.code) {
                let (s, d, c) = (s as usize, d as usize, c as usize);
                let (hs, tc) = (&hn[s * h..(s + 1) * h], &layer.table[c * h..(c + 1) * h]);
                for ((a, &p), &q) in agg[d * h..(d + 1) * h].iter_mut().zip(hs).zip(tc) {
                    *a += (p + q).max(0.0);
                }
            }
            for (v, &div) in batch.divisor.iter().enumerate() {
                let div = div as f32;
                for j in v * h..(v + 1) * h {
                    agg[j] = agg[j] / div + layer.one_eps * hn[j];
                }
            }
            let t = linear(&linear(&agg, n, &layer.mlp0, true), n, &layer.mlp2, false);
            for (xv, &u) in x.iter_mut().zip(&t) {
                *xv = (u + *xv).max(0.0);
            }
            let normed = layer_norm(&x, h, &self.final_norm);
            for v in 0..n {
                emb[v * wide + i * h..v * wide + (i + 1) * h]
                    .copy_from_slice(&normed[v * h..(v + 1) * h]);
            }
        }
        let rows: Vec<f32> = batch
            .legal
            .iter()
            .flat_map(|&l| &emb[l as usize * wide..(l as usize + 1) * wide])
            .copied()
            .collect();
        let logits = linear(
            &linear(&rows, batch.legal.len(), &self.policy0, true),
            batch.legal.len(),
            &self.policy2,
            false,
        );

        let mut pooled = vec![0f32; g * 2 * wide];
        for chunk in pooled.chunks_mut(2 * wide) {
            chunk[wide..].fill(f32::MIN);
        }
        for v in 0..n {
            let gi = batch.node_graph[v] as usize;
            let row = &emb[v * wide..(v + 1) * wide];
            let (mean, max) = pooled[gi * 2 * wide..(gi + 1) * 2 * wide].split_at_mut(wide);
            if batch.mean_mask[v] != 0 {
                mean.iter_mut().zip(row).for_each(|(m, &e)| *m += e);
            }
            if batch.real[v] != 0 {
                max.iter_mut().zip(row).for_each(|(m, &e)| *m = m.max(e));
            }
        }
        for (chunk, &den) in pooled.chunks_mut(2 * wide).zip(&batch.mean_den) {
            chunk[..wide].iter_mut().for_each(|m| *m /= den as f32);
        }
        let bins = linear(&linear(&pooled, g, &self.fc1, true), g, &self.fc2, false);
        let value = bins
            .chunks(self.fc2.n)
            .map(|row| {
                let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = row.iter().map(|&b| (b - top).exp()).collect();
                let sum: f32 = ex.iter().sum();
                ex.iter()
                    .zip(&self.support)
                    .map(|(&e, &s)| e / sum * s)
                    .sum::<f32>()
                    .clamp(-1.0, 1.0)
            })
            .collect();

        let mut top = vec![f32::NEG_INFINITY; g];
        for (&lg, &gi) in logits.iter().zip(&batch.legal_graph) {
            top[gi as usize] = top[gi as usize].max(lg);
        }
        let mut probs: Vec<f32> = logits
            .iter()
            .zip(&batch.legal_graph)
            .map(|(&lg, &gi)| (lg - top[gi as usize]).exp())
            .collect();
        let mut sums = vec![0f32; g];
        for (&p, &gi) in probs.iter().zip(&batch.legal_graph) {
            sums[gi as usize] += p;
        }
        for (p, &gi) in probs.iter_mut().zip(&batch.legal_graph) {
            *p /= sums[gi as usize];
        }
        Ok(NetOutputs {
            logits,
            probs,
            value,
        })
    }
}

/// `a [m × k] · wt [k × n] + b`, in row blocks of 4 so each weight row is read once a block.
fn linear(a: &[f32], m: usize, layer: &Linear, relu: bool) -> Vec<f32> {
    let (k, n) = (layer.k, layer.n);
    let mut out = vec![0f32; m * n];
    for r in (0..m).step_by(4) {
        let rows = (m - r).min(4);
        for i in 0..rows {
            out[(r + i) * n..(r + i + 1) * n].copy_from_slice(&layer.b);
        }
        for kk in 0..k {
            let w = &layer.wt[kk * n..(kk + 1) * n];
            for i in 0..rows {
                let av = a[(r + i) * k + kk];
                if av != 0.0 {
                    out[(r + i) * n..(r + i + 1) * n]
                        .iter_mut()
                        .zip(w)
                        .for_each(|(o, &wv)| *o += av * wv);
                }
            }
        }
        if relu {
            out[r * n..(r + rows) * n]
                .iter_mut()
                .for_each(|v| *v = v.max(0.0));
        }
    }
    out
}

/// LayerNorm over each row of `h`, eps 1e-5, as torch's.
#[allow(clippy::cast_precision_loss)] // a hidden width is far below 2^23
fn layer_norm(x: &[f32], h: usize, norm: &Norm) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (row, dst) in x.chunks(h).zip(out.chunks_mut(h)) {
        let mean = row.iter().sum::<f32>() / h as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (j, o) in dst.iter_mut().enumerate() {
            *o = (row[j] - mean) * inv * norm.w[j] + norm.b[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-layer net of width 2 over 3 features and 2 codes, every float 0.1, laid out as the exporter writes it.
    fn tiny() -> (Vec<TensorRow>, Vec<f32>) {
        let shapes: [(&str, usize, usize); 17] = [
            ("input_proj.wt", 3, 2),
            ("input_proj.b", 1, 2),
            ("norm0.w", 1, 2),
            ("norm0.b", 1, 2),
            ("table0", 2, 2),
            ("eps0", 1, 1),
            ("mlp0.0.wt", 2, 4),
            ("mlp0.0.b", 1, 4),
            ("mlp0.2.wt", 4, 2),
            ("mlp0.2.b", 1, 2),
            ("final_norm.w", 1, 2),
            ("final_norm.b", 1, 2),
            ("policy.0.wt", 2, 3),
            ("policy.0.b", 1, 3),
            ("policy.2.wt", 3, 1),
            ("policy.2.b", 1, 1),
            ("fc1.wt", 4, 2),
        ];
        let tail: [(&str, usize, usize); 3] = [("fc1.b", 1, 2), ("fc2.wt", 2, 3), ("fc2.b", 1, 3)];
        let mut table = Vec::new();
        let mut offset = 0;
        for (name, rows, cols) in shapes.iter().chain(&tail).chain(&[("support", 1, 3)]) {
            table.push(TensorRow {
                name: (*name).to_owned(),
                rows: *rows,
                cols: *cols,
                offset,
            });
            offset += rows * cols;
        }
        (table, vec![0.1; offset])
    }

    fn batch() -> WebBatch {
        WebBatch {
            n_graphs: 1,
            x: vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5],
            src: vec![0, 1],
            dst: vec![1, 0],
            code: vec![0, 1],
            legal: vec![1],
            node_graph: vec![0, 0],
            legal_graph: vec![0],
            divisor: vec![1, 1],
            real: vec![1, 1],
            mean_mask: vec![1, 0],
            mean_den: vec![1],
            leaves: Vec::new(),
        }
    }

    #[test]
    fn a_served_table_loads_and_a_graphs_probs_sum_to_one() {
        let (table, floats) = tiny();
        let net = RustForward::load(&table, &floats, 3).expect("the tiny table is one served net");
        let out = net.forward(&batch()).expect("a well-formed batch");
        assert_eq!(
            (out.logits.len(), out.probs.len(), out.value.len()),
            (1, 1, 1)
        );
        assert!((out.probs[0] - 1.0).abs() < 1e-6 && out.value[0].abs() <= 1.0);
    }

    #[test]
    fn a_table_and_a_file_that_disagree_are_refused_by_name() {
        let (table, floats) = tiny();
        assert!(matches!(
            RustForward::load(&table, &floats, 4),
            Err(ForwardError::Shape { .. })
        ));
        assert!(matches!(
            RustForward::load(&table, &floats[1..], 3),
            Err(ForwardError::OutOfFile { .. })
        ));
        let no_fc2: Vec<TensorRow> = table
            .iter()
            .filter(|r| r.name != "fc2.wt")
            .cloned()
            .collect();
        assert!(
            matches!(RustForward::load(&no_fc2, &floats, 3), Err(ForwardError::Missing(n)) if n == "fc2.wt")
        );
        let mut narrow = table.clone();
        for r in narrow
            .iter_mut()
            .filter(|r| r.name == "fc2.wt" || r.name == "fc2.b" || r.name == "support")
        {
            r.cols = 0;
        }
        assert!(
            matches!(RustForward::load(&narrow, &floats, 3), Err(ForwardError::Shape { name, .. }) if name == "fc2.wt")
        );
    }

    #[test]
    fn a_batch_whose_index_or_length_its_shape_does_not_allow_is_refused() {
        let (table, floats) = tiny();
        let net = RustForward::load(&table, &floats, 3).expect("the tiny table is one served net");
        let mutations: [fn(&mut WebBatch); 5] = [
            |b| b.src[0] = 2,
            |b| b.code[0] = 2,
            |b| b.legal_graph[0] = 1,
            |b| b.divisor[0] = 0,
            |b| {
                b.real.pop();
            },
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut b = batch();
            mutate(&mut b);
            assert!(
                matches!(net.forward(&b), Err(ForwardError::Batch(_))),
                "mutation {i}"
            );
        }
    }
}
