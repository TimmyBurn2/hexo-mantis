# 00_MAP — slimming census, the BEFORE figure and the partition

Tree: `origin/dev` = `69e153296c6d8245456d08ab5ddb6a0749e47c67` (2026-09-21 22:53 +0200), 1 071 tracked paths.
Census branch: `claude/slim-scout-census-v3i2hj` (the platform's branch; the packet's `slim` is this one).
Register at HEAD: `### R358` present; the head entry is R367 (the packet snapshot said R358 — verified newer).

## 1. Size (BEFORE)

Unit: text lines from `git diff --numstat <empty-tree> HEAD` (git's own count; binary files count 0 and are
tallied separately). `wc -l` is NOT used: it counts newline bytes inside the 33 binary fixtures.

Command (per top dir):

```
E=$(git hash-object -t tree /dev/null)
git diff --numstat $E HEAD | awk -F'\t' '{p=$3; top=(index(p,"/")?substr(p,1,index(p,"/")-1):"<root>"); F[top]++; if($1=="-")B[top]++; else L[top]+=$1} END{for(k in F) printf "%-10s %5d %4d %7d\n",k,F[k],B[k]+0,L[k]}' | sort -k4 -n -r
```

| top dir | files | binary | text lines | of which .py | .rs | .md | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| tests | 546 | 33 | 279 440 | 97 375 | 0 | 36 | 182 029 (fixture data) |
| docs | 67 | 0 | 40 710 | 0 | 0 | 40 710 | 0 |
| crates | 145 | 0 | 40 012 | 0 | 39 019 | 0 | 993 |
| src | 193 | 0 | 38 133 | 37 201 | 0 | 0 | 932 |
| tools | 98 | 0 | 16 362 | 14 337 | 0 | 29 | 1 996 |
| (root files) | 13 | 0 | 3 646 | 0 | 0 | 303 | 3 343 |
| configs | 6 | 0 | 1 055 | | | | 1 055 |
| vendor | 2 | 0 | 149 | | | | 149 |
| .github | 1 | 0 | 141 | | | | 141 |
| **all** | **1 071** | **33** | **419 648** | 148 913 | 39 019 | 41 078 | 190 638 |

Command (per package): the same pipe with the key `src/mantis/<pkg>`, `crates/<crate>`, `tests/<pkg>`,
`tools/<pkg>`, `docs/<dir>` (files directly under a top dir keyed `<top>/<files>`):

```
git diff --numstat $E HEAD | awk -F'\t' '{p=$3; c=split(p,a,"/"); if(c==1)k="<root>"; else if(a[1]=="src"&&c>=4)k=a[1]"/"a[2]"/"a[3]; else if(a[1]=="src")k="src/mantis/<files>"; else if((a[1]=="crates"||a[1]=="tests"||a[1]=="tools"||a[1]=="docs")&&c>=3)k=a[1]"/"a[2]; else k=a[1]"/<files>"; F[k]++; if($1=="-")B[k]++; else L[k]+=$1} END{for(k in F) printf "%-32s %5d %4d %7d\n",k,F[k],B[k]+0,L[k]}' | sort
```

```
package                          files  bin   lines
.github/<files>                      1    0     141
<root>                              13    0    3646
configs/<files>                      6    0    1055
crates/mantis-bridge                13    0    5176
crates/mantis-core                  17    0    3855
crates/mantis-encoding               9    0    1647
crates/mantis-graph                  8    0    2837
crates/mantis-search                35    0   11382
crates/mantis-selfplay              63    0   15115
docs/audits                          5    0    3096
docs/contracts                       9    0    1502
docs/design                         43    0   12314
docs/governance                     10    0   23798
src/mantis/<files>                   4    0    1695
src/mantis/arena                     9    0     934
src/mantis/bots                      6    0     860
src/mantis/config                   45    0    5273
src/mantis/data                     14    0    2529
src/mantis/deploy                    1    0       1
src/mantis/diagnostics              11    0    4154
src/mantis/encoding                  7    0    1514
src/mantis/env                       2    0      84
src/mantis/eval                     11    0    3110
src/mantis/model                    10    0    1358
src/mantis/monitor                  12    0    2623
src/mantis/selfplay                 14    0    4356
src/mantis/train                    38    0    9157
src/mantis/util                      9    0     485
tests/<files>                       24    0    6186
tests/arena                         11    0    1556
tests/bots                           5    0    1543
tests/bridge                        16    0    1683
tests/config                        56    0   11050
tests/data                           7    0    1099
tests/diagnostics                   19    0    4274
tests/encoding                      15    0    1437
tests/env                            1    0     113
tests/eval                          47    0    8851
tests/fixtures                      64   33  182119
tests/model                         32    0    7917
tests/monitor                       18    0    3905
tests/selfplay                      57    0   10339
tests/tools                         73    0   14286
tests/train                         95    0   22428
tests/util                           6    0     654
tools/<files>                       23    0    4972
tools/analyzer                      13    0    1305
tools/ci_gates                      21    0    5569
tools/config_templates               1    0     247
tools/dashboard                     17    0    2144
tools/ladder                        10    0     996
tools/probe1                         8    0     750
tools/viewer                         5    0     379
vendor/<files>                       2    0     149
```

## 2. Partition (area slices)

Every `git ls-files` path is matched against the 21 slice regexes below; exactly one must match.
The awk program is this block, verbatim; the command extracts it from this file and runs it:

```
sed -n '/^```awk$/,/^```$/p' docs/slim/00_MAP.md | sed '1d;$d' > /tmp/partition.awk
git diff --numstat $(git hash-object -t tree /dev/null) HEAD | awk -F'\t' -f /tmp/partition.awk
```

```awk
BEGIN {
  n = 0
  S[++n] = "R1";  P["R1"] = "^crates/mantis-search/"
  S[++n] = "R2";  P["R2"] = "^crates/mantis-selfplay/"
  S[++n] = "R3";  P["R3"] = "^crates/mantis-(core|encoding|graph|bridge)/"
  S[++n] = "C1";  P["C1"] = "^src/mantis/(train|model|encoding)/"
  S[++n] = "C2";  P["C2"] = "^src/mantis/((selfplay|data|config|env)/|[^/]+$)"
  S[++n] = "C3";  P["C3"] = "^src/mantis/(eval|arena|bots|diagnostics|monitor|util|deploy)/"
  S[++n] = "L1";  P["L1"] = "^(tools/(ci_gates|config_templates)/|Makefile$|\\.github/)"
  S[++n] = "L2";  P["L2"] = "^tools/([^/]+$|analyzer/|dashboard/|ladder/|probe1/|viewer/)"
  S[++n] = "T1";  P["T1"] = "^tests/train/([^t]|t[^e]|te[^s]|tes[^t]|test[^_]|test_[a-g])"
  S[++n] = "T2";  P["T2"] = "^tests/train/test_[h-z]"
  S[++n] = "T3";  P["T3"] = "^tests/tools/"
  S[++n] = "T4";  P["T4"] = "^tests/(config|data|encoding|env|util)/"
  S[++n] = "T5";  P["T5"] = "^tests/(selfplay|bridge|arena)/"
  S[++n] = "T6";  P["T6"] = "^tests/(eval|bots|diagnostics)/"
  S[++n] = "T7";  P["T7"] = "^tests/([^/]+$|monitor/)"
  S[++n] = "T8";  P["T8"] = "^tests/model/"
  S[++n] = "T9";  P["T9"] = "^tests/fixtures/"
  S[++n] = "D1";  P["D1"] = "^docs/(governance/[^/]+$|audits/|contracts/|slim/)"
  S[++n] = "D2";  P["D2"] = "^docs/governance/archive/"
  S[++n] = "D3";  P["D3"] = "^docs/design/"
  S[++n] = "D4";  P["D4"] = "^(configs/|vendor/|(\\.gitattributes|\\.gitignore|CLAUDE\\.md|Cargo\\.lock|Cargo\\.toml|LICENSE|README\\.md|mise\\.toml|pyproject\\.toml|rust-toolchain\\.toml|rustfmt\\.toml|uv\\.lock)$)"
}
{
  add = $1; path = $3; hits = 0; who = ""
  for (i = 1; i <= n; i++) if (path ~ P[S[i]]) { hits++; who = S[i] }
  if (hits == 0) { unassigned++; print "UNASSIGNED " path > "/dev/stderr"; next }
  if (hits > 1) { double++; print "DOUBLE " path > "/dev/stderr"; next }
  F[who]++
  if (add == "-") B[who]++; else L[who] += add
}
END {
  printf "%-4s %6s %6s %8s  %s\n", "id", "files", "binary", "lines", "pattern"
  for (i = 1; i <= n; i++) { s = S[i]; tot += L[s]; ft += F[s]; printf "%-4s %6d %6d %8d  %s\n", s, F[s], B[s], L[s], P[s] }
  printf "ALL  %6d %8s %8d\n", ft, "", tot
  printf "partition check: unassigned %d, double-assigned %d\n", unassigned + 0, double + 0
}
```

Result at `69e1532`:

```
id    files binary    lines  pattern
R1       35      0    11382  ^crates/mantis-search/
R2       63      0    15115  ^crates/mantis-selfplay/
R3       47      0    13515  ^crates/mantis-(core|encoding|graph|bridge)/
C1       55      0    12029  ^src/mantis/(train|model|encoding)/
C2       79      0    13937  ^src/mantis/((selfplay|data|config|env)/|[^/]+$)
C3       59      0    12167  ^src/mantis/(eval|arena|bots|diagnostics|monitor|util|deploy)/
L1       24      0     6043  ^(tools/(ci_gates|config_templates)/|Makefile$|\.github/)
L2       76      0    10546  ^tools/([^/]+$|analyzer/|dashboard/|ladder/|probe1/|viewer/)
T1       46      0    12031  ^tests/train/([^t]|t[^e]|te[^s]|tes[^t]|test[^_]|test_[a-g])
T2       49      0    10397  ^tests/train/test_[h-z]
T3       73      0    14286  ^tests/tools/
T4       85      0    14353  ^tests/(config|data|encoding|env|util)/
T5       84      0    13578  ^tests/(selfplay|bridge|arena)/
T6       71      0    14668  ^tests/(eval|bots|diagnostics)/
T7       42      0    10091  ^tests/([^/]+$|monitor/)
T8       32      0     7917  ^tests/model/
T9       64     33   182119  ^tests/fixtures/
D1       20      0    11180  ^docs/(governance/[^/]+$|audits/|contracts/|slim/)
D2        4      0    17216  ^docs/governance/archive/
D3       43      0    12314  ^docs/design/
D4       20      0     4764  ^(configs/|vendor/|(\.gitattributes|\.gitignore|CLAUDE\.md|Cargo\.lock|Cargo\.toml|LICENSE|README\.md|mise\.toml|pyproject\.toml|rust-toolchain\.toml|rustfmt\.toml|uv\.lock)$)
ALL    1071            419648
partition check: unassigned 0, double-assigned 0
```

Slice notes:
- Split rule (~15k): R2 (15 115) is one crate and within tolerance. T9 (182 119) is over by DATA: 167 362 of
  its lines are three fixture files (`graph_parity/wpa_positions.json` 124 769, `graph_parity/manifest.tsv`
  23 771, `mctx_parity/mctx_parity_v1.json` 18 822) inside indivisible packages; its code/manifest part is small.
  D2 (17 216) is four frozen files in one directory. tests/train had no sub-package, so T1/T2 split it at the
  alphabetical midpoint (`test_[a-g]*` + underscore helpers | `test_[h-z]*`) — a deviation from "package
  boundaries", recorded.
- `docs/slim/` is placed in D1 so the partition still holds after this census commits.

| slice | scout file | area | content |
|---|---|---|---|
| R1 | S-A-RUST-1.md | A-RUST | crates/mantis-search |
| R2 | S-A-RUST-2.md | A-RUST | crates/mantis-selfplay |
| R3 | S-A-RUST-3.md | A-RUST | crates/mantis-core, -encoding, -graph, -bridge |
| C1 | S-A-CORE-1.md | A-CORE | src/mantis/train, model, encoding |
| C2 | S-A-CORE-2.md | A-CORE | src/mantis/selfplay, data, config, env + package root files |
| C3 | S-A-CORE-3.md | A-CORE | src/mantis/eval, arena, bots, diagnostics, monitor, util, deploy |
| L1 | S-A-TOOLS-1.md | A-TOOLS | tools/ci_gates, tools/config_templates, Makefile, .github |
| L2 | S-A-TOOLS-2.md | A-TOOLS | tools/ top-level modules, analyzer, dashboard, ladder, probe1, viewer |
| T1 | S-A-TESTS-1.md | A-TESTS | tests/train a–g + helpers |
| T2 | S-A-TESTS-2.md | A-TESTS | tests/train h–z |
| T3 | S-A-TESTS-3.md | A-TESTS | tests/tools |
| T4 | S-A-TESTS-4.md | A-TESTS | tests/config, data, encoding, env, util |
| T5 | S-A-TESTS-5.md | A-TESTS | tests/selfplay, bridge, arena |
| T6 | S-A-TESTS-6.md | A-TESTS | tests/eval, bots, diagnostics |
| T7 | S-A-TESTS-7.md | A-TESTS | tests/ root modules + tests/monitor |
| T8 | S-A-TESTS-8.md | A-TESTS | tests/model (incl. the conformance suite) |
| T9 | S-A-TESTS-9.md | A-TESTS | tests/fixtures |
| D1 | S-A-DOCS-1.md | A-DOCS | docs/governance (non-archive), docs/audits, docs/contracts |
| D2 | S-A-DOCS-2.md | A-DOCS | docs/governance/archive (frozen) |
| D3 | S-A-DOCS-3.md | A-DOCS | docs/design/** |
| D4 | S-A-DOCS-4.md | A-DOCS | configs/, vendor/, root files |
| — | S-L-DUP.md, S-L-SEAM.md, S-L-STYLE.md | LENS | whole tree |
