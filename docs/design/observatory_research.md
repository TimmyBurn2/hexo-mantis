# OBSERVATORY research — how the reference tools read a game beside an engine, and lay out a run (2026-09-14)

Companion to `docs/design/observatory_design.md`. The question: how lichess, chess.com, OGS,
Sabaki, Lizzie, KaTrain, KataGo's analysis engine, TensorBoard and Red Blob Games' hex-grid guide
solve (a) the layout of board vs move list vs evaluation on a desk and on a phone, (b) showing the
engine's evaluation beside the position, (c) keyboard and touch, (d) absence and uncertainty,
(e) deep-linking a position, (f) performance at scale — precisely enough to say what a merged,
offline, phone-usable observatory takes and rejects, with the reason.

Rule: every claim carries its source in parentheses — a URL, and for code `path::symbol` at the
repository's default branch as fetched on 2026-09-14. "unverified" marks anything not confirmed
from a primary source; "(via fetch summary)" marks a page read only through a summariser. Sources
were read by a research agent of this session (the raw notes and the fetched source files are in
that session's scratchpad, untracked); the takes and rejects at the end are this packet's
decisions, each with the house rule that decided it.

Repositories and branches read: lichess-org/lila `master`; online-go/online-go.com `master`;
online-go/goban `main`; SabakiHQ/Sabaki `master`; SabakiHQ/Shudan `master`; featurecat/lizzie
`master`; sanderland/katrain `master`; lightvector/KataGo `master`; tensorflow/tensorboard `master`.

---

## 1. lichess analysis board (lichess-org/lila)

### (a) Layout
- The analysis page is one CSS grid. Single column (phone, `mq-is-col1` is the default with no media query): rows `'board' 'controls' 'kb-move' 'tools' 'side' 'round-training' 'under' 'chat' 'uchat'` — board on top, then the controls bar, then tools (engine + move list), then everything else (`ui/analyse/css/_layout.scss::.analyse grid-template-areas`).
- Two columns (`mq-at-least-col2`): columns `board | gauge | tools`; areas `'board gauge tools' / 'kb-move . controls' / 'under . controls' / ...` — the eval gauge is a dedicated narrow grid column between board and tools, and the computer-analysis chart lives in `under` beneath the board (`ui/analyse/css/_layout.scss` `@include mq-at-least-col2`).
- Three columns (`mq-at-least-col3`): `'side . board gauge tools'` — side panel left, board centre, gauge, tools right (`ui/analyse/css/_layout.scss` `@include mq-at-least-col3`).
- Gauge column width is `$block-gap` normally and `calc($block-gap + 4px*2)` when `.gauge-on` (`ui/analyse/css/_layout.scss::---analyse-gauge-col`); the tools column height is tied to the board height (`&__tools { height: var(---cg-height, 100%) }`).
- Phone: the gauge is `display: none` on col1 both in the layout and in the gauge's own stylesheet (`ui/analyse/css/_layout.scss::.eval-gauge{display:none}`; `ui/lib/css/ceval/_eval-gauge.scss` `@include mq-is-col1 { display: none; }`); the ceval "pearl" (big eval number) and the on/off toggle are hidden inside the tools box (`ui/analyse/css/_tools-mobile.scss::.ceval .cmn-toggle, .ceval pearl { display: none; }`) and the eval number is instead shown inside a tab button of the controls bar (`ui/analyse/src/view/controls.ts::renderMobileCevalTab` → `hl('eval', evalstr)` when `displayColumns() === 1`).
- Phone tools ordering is forced by flex `order`: sub-box 0, ceval/pv_box/fork 1, move list 2 (`ui/analyse/css/_tools-mobile.scss::.analyse__tools`).
- The computer-analysis chart is not in the initial DOM; it is a canvas injected into the `computer-analysis` underboard panel on demand and the chart module is lazy-loaded (`ui/analyse/src/serverSideUnderboard.ts` lines 73–82: `'<div id="acpl-chart-container"><canvas id="acpl-chart"></canvas></div>'`, `site.asset.loadEsm<ChartGame>('chart.game').then(m => m.acpl(...))`).

### (b) Evaluation beside the position
- Eval → winning chances: `2 / (1 + exp(-0.00368208 * cp)) - 1`, cp clamped to ±1000; mate is mapped to cp `(21 - min(10,|mate|)) * 100` with sign (`ui/lib/src/ceval/winningChances.ts::rawWinningChances`, `cpWinningChances`, `mateWinningChances`; constant attributed to lila PR 11148 in the file comment).
- Gauge: 7 ticks at `(i+1)*12.5%` heights with the fourth marked `.zero`; the fill is `--eval-percent: ${100 - (ev + 1) * 50}%` where `ev = povChances('white', bestEv)`; when orientation is black the whole gauge gets `.reverse` (`rotateX(180deg)`) (`ui/lib/src/ceval/view/main.ts::renderGauge`; `ui/lib/css/ceval/_eval-gauge.scss`). The fill is a two-stop `linear-gradient` split at `--eval-percent`, animated with `transition: --eval-percent 1s ease` (`_eval-gauge.scss`).
- Best eval preference: local ceval on the node beats server eval; the server eval is used only if `showEvaluation()` (`ui/lib/src/ceval/view/main.ts::getBestEval`).
- Ceval box: pearl = cp rendered via `renderEval`, or `#N` for mate; engine name; depth text `depthX(depth)` with `/99` when infinite/deeper; knps; cloud badge when `client.cloud` (`ui/lib/src/ceval/view/main.ts::renderCeval`, `localEvalNodes`, `localInfo`). A progress `.bar` spans `percent` of the search budget (depth, nodes, or movetime) (`renderCeval` lines ~176–182).
- MultiPV lines: `renderPvs` renders `multiPv` rows each with its eval and up to `MAX_NUM_MOVES` SAN moves; hovering a PV move shows a small preview board (`renderPvBoard`) and the scroll wheel steps through the hovered PV (`ui/lib/src/ceval/view/main.ts::renderPvs` wheel handler, `renderPvBoard`).
- Arrows on the board: server best move in `paleGreen`; the local best PV drawn as a chain of `paleBlue` arrows (`drawManeuver`); every other multiPV first move drawn `paleGrey` with `lineWidth: Math.round(12 - shift * 50)` ("12 to 2") where `shift = povDiff(color, pvs[0], pv)` and only if `0 <= shift < 0.2`; threat mode draws the opponent's PV in `red`/`paleRed` (`ui/analyse/src/autoShape.ts::compute`, lines 130–160). povDiff is `(povChances(e1) - povChances(e2)) / 2` (`winningChances.ts::povDiff`).
- Per-move graph ("computer analysis" chart): Chart.js line of `winningChances.povChances('white', {cp})` per ply — "Plot winchance because logarithmic but display the corresponding cp.eval from AnalyseData in the tooltip"; filled above origin white / below black; y-axis fixed to ±1.05 with ticks hidden; x-axis hidden; a vertical orange `plyLine` marks the current ply and is updated on the `ply` pubsub event; clicking emits `analysis.chart.click` with the index; blunder/mistake/inaccuracy glyphs colour hover points `#db3031`/`#e69d00`/`#4da3d5` (`ui/chart/src/acpl.ts` makeDataset/config/glyphProperties; `ui/chart/src/index.ts::axisOpts`, `chartYMax = 1.05`, `plyLine`, `selectPly`).
- No ownership/heat map exists in lila's analysis board (nothing found in `ui/analyse/src` or `ui/lib/src/ceval`; absence unverified beyond the file listing).

### (c) Keyboard and touch
- Bindings (`ui/analyse/src/keyboard.ts::bind`): `left`/`k` prev; `right`/`j` next; `up`/`0`/`home` first (ArrowUp selects previous fork if one exists); `down`/`$`/`end` last (ArrowDown selects next fork); `shift+left|k` previousBranch; `shift+right|j` nextBranch; `shift+up|down` stepLine prev/next; `shift+c` toggle comments; `shift+i` inline notation; `space` play best move (or enable ceval if allowed and off); `h` action menu; `f` flip; `b` board editor; `?` keyboard help; `l` toggle local ceval; `z` toggle showStaticAnalysis; `a` best-move arrows; `v` variation arrows; `x` threat mode; `e` explorer; `shift+space` first explorer move; tapping `Shift` alone cycles forks; tapping `Ctrl` alone toggles disclosure (`addModifierKeyListeners`).
- The `?` overlay is server HTML at `/analysis/help` and lists the same keys plus "Mouse tricks": "Scrolling over the board navigates through the game", "Scrolling over computer variations allows previewing them", "Right-click (or shift+click) draws circles and arrows" (https://lichess.org/analysis/help, reachable, via fetch summary).
- Wheel over the board steps moves via `stepwiseScroll`, but only when `'ontouchstart' in window` is false and the `scrollMoves` preference is on (`ui/analyse/src/view/components.ts::renderBoard`).
- Touch: the controls bar's prev/next buttons repeat while held (`ui/analyse/src/view/controls.ts::holdControl` → `repeater`), and the bar sets `touch-action: pan-y` (`_tools-mobile.scss::.analyse__controls`). No swipe-on-board binding was found in `ui/analyse/src` (absence unverified).

### (d) Absence and uncertainty
- Gauge with no eval: keeps the last value (`gaugeLast`) and adds class `empty` → `opacity: 0.5` (`ui/lib/src/ceval/view/main.ts::renderGauge`; `_eval-gauge.scss::&.empty`).
- Pearl states with no score: engine off → empty icon; game over / threefold → `'-'`; `CevalState.Failed` → red caution icon; otherwise a spinner `icon.ddloader` (`renderCeval`). Progress bar `percent` is 0 while computing with nothing, 100 on game over.
- Chart tooltip returns `''` when the node has no eval ("Pos is mate") (`ui/chart/src/acpl.ts` tooltip label callback); partial server analysis keeps the chart loader visible until `!d.analysis.partial` (`ui/analyse/src/serverSideUnderboard.ts` line 63).
- Confidence is conveyed as depth (`depthX`), knps and the cloud badge — never as a numeric interval (`localInfo`, `localEvalNodes`).

### (e) Deep-linking
- URL hash `#<ply>` or `#last` sets the initial mainline position; the hash is then removed with `history.replaceState` (`ui/analyse/src/ctrl.ts::makeInitialPath` lines 270–277). While browsing, `updateHref` writes `'#' + node.ply` back, debounced 750 ms, except in studies (`ctrl.ts::updateHref`). Other hashes: `#keyboard` opens help, `#practice`, `#menu` (`ctrl.ts` lines 116, 192–193). Query `?evals=0` disallows ceval; `?engine=` selects an engine (`ctrl.ts::isCevalAllowed`, line 197).

### (f) Performance at scale
- Chart module lazy-loaded on panel open (`serverSideUnderboard.ts` `loadEsm('chart.game')`); server analysis streams in with `analysis.partial` and the chart is updated in place via `updateData` (`serverSideUnderboard.ts::onServerProgress`, `acpl.ts::acplChart.updateData`).
- Evals are cached per FEN in the client and pushed to the shared cloud only when depth ≥ 20 and nodes ≥ 3e6 (`ui/analyse/src/evalCache.ts::evalPutMinDepth = 20`, `evalPutMinNodes = 3e6`); local evals are persisted to IndexedDB (`ctrl.ts` `this.idbTree.saveCeval(path, ev)`).
- The gauge tick VNodes are built once and reused (`renderGauge` `gaugeTicks ??= ...`).

---

## 2. chess.com Game Review / analysis board (help centre only)

All chess.com items are from support.chess.com articles read via fetch summary; the site publishes no source.

### (a) Layout
- App: "The moves display below the board"; navigate with "arrows at either side of the move list below the board to browse moves one at a time", or by "Swiping left or right on the move list"; a green "Next" button jumps "to the next key move in the game"; a coach speech bubble updates each move and can be dragged down (https://support.chess.com/en/articles/10328363-how-do-i-use-game-review-on-the-app, via fetch summary).
- Web analysis board: an "evaluation bar" on the left that updates as you move; an "Explore tab in the right-side panel"; the three-dots menu next to the Analysis toggle can "enable or disable the evaluation bar, engine lines, suggestion arrows on your board and move feedback" (https://support.chess.com/en/articles/8583825-how-do-i-use-the-analysis-board, via fetch summary). Exact desktop placement of graph vs move list is not stated (unverified).

### (b) Evaluation beside the position
- Game Review shows "a graph [that] indicates who had the advantage at each stage of the game", an accuracy score 0–100, and a per-move classification label: Brilliant, Great, Best, Excellent, Good, Book, Inaccuracy, Mistake, Miss, Blunder (https://support.chess.com/en/articles/8584089-how-does-game-review-work, via fetch summary).
- Classification is by an "Expected Points Model" (1.00 certain win, 0.50 equal) with bands of expected points lost: Best 0.00, Excellent 0.00–0.02, Good 0.02–0.05, Inaccuracy 0.05–0.10, Mistake 0.10–0.20, Blunder 0.20–1.00; Brilliant/Great/Miss are rating-dependent special cases (https://support.chess.com/en/articles/8572705-how-are-moves-classified-what-is-a-blunder-or-brilliant-etc, via fetch summary).
- Accuracy is "CAPS2"; the formula is withheld ("the math on how these are calculated has changed"); target band 50–95 for most players (https://support.chess.com/en/articles/8708970-how-is-accuracy-in-analysis-determined, via fetch summary).
- Move-list colouring setting: "Move Strength Coloring for All Moves" vs "for Key Moves" under Settings > Interface > Analysis (https://support.chess.com/en/articles/8648715-how-do-i-change-what-kinds-of-moves-are-highlighted-in-analysis, via fetch summary).
- Engines: server-side "Stockfish 18" with NNUE for Game Review / Cloud Analysis; browser users get a "Lite" handcrafted-eval build unless members; "diminishing returns in engine quality as depth gets past the low 20's"; members can set max depth to 99 (https://support.chess.com/en/articles/9462780-how-do-the-chess-engines-on-chess-com-work, via fetch summary).

### (c) Keyboard / touch
- Only the app touch model above is documented (arrows, swipe on the move list, Next button). "Hotkeys usage" is a toggle in the Interface tab but the keys are not listed in the fetched article (https://support.chess.com/en/articles/8583825..., via fetch summary). Keyboard bindings: unverified.

### (d) Absence / uncertainty
- "When you open Game Review, Chess.com performs a more detailed analysis, which can result in different move classifications" — classifications are provisional until the server review runs (https://support.chess.com/en/articles/11845102-why-did-my-move-classification-change-in-game-review, via search-result summary; not read in full).
- No statement on how an unanalysed position is drawn (unverified).

### (e) Deep-linking — not documented in the help centre (unverified).

### (f) Performance / paywall
- Free members get "key insights like Accuracy, move classifications, and Self Analysis" and a limited number of daily reviews (count not extracted — unverified); Diamond gets unlimited reviews, Cloud Analysis, and "Deep and Maximum" engine strength (https://support.chess.com/en/articles/8584089..., https://support.chess.com/en/articles/8708936-what-is-analysis-depth-what-does-it-mean, via fetch summary).

---

## 3. OGS (online-go/online-go.com + online-go/goban)

### (a) Layout
- The game page is `GobanView`: a flex row, the board centre and a sidebar (`src/components/GobanView/GobanView.css::.GobanView { display:flex; flex-direction:row }`). View mode is chosen by aspect ratio: `portrait` when `(w/h <= 0.8 || w < bar_width*2) && w < 1280`, else wide/square (`src/components/GobanView/util.ts::goban_view_mode` lines 74–79); `goban_view_squashed()` is `innerHeight <= 500` (`util.ts` line 105).
- Landscape: sidebar width defaults to "half of the space the board does not use"; the root is a size container (`container-type: size`) so the board is height-limited (`GobanView.css` comments, lines ~64–75).
- Portrait: `flex-direction: column`; a single scroll container `.GobanView-mobile-scroll` holds the board "stage" (opponent card above, player card + play buttons below, board shrinks to keep both slots on screen, `min-height: 8rem`) and then the tab panels (`.GobanView-mobile-panels`) so "the user can scroll the board off-screen to reach the content below it"; an opt-in `has-portrait-split` pins the board and scrolls the panels with a drag handle (`GobanView.css` `&.portrait`, `&.has-portrait-split`; `GobanView.tsx` props `aboveBoard`, `belowBoard`, `portraitSplit`, `leftAside` doc comments).
- AI review panel content: review selector + win-rate/score readout, the D3 `ReviewChart`, a `WorstMovesList` (6 entries), the score/winrate toggle, and a `SummaryTable` (KataGo only) (`src/components/AIReview/AIReview.tsx` lines 640–710; `WORST_MOVES_SHOWN = 6` line 60).

### (b) Evaluation beside the position
- Data model (`goban/src/engine/formats/JGOF.ts::JGOFAIReview`, `JGOFAIReviewMove`, `JGOFAIReviewMoveVariation`): review-level `win_rates?: number[]` ("predicted probability that black will win for all moves"), `scores?: number[]`, `strength: number`, `type: "fast" | "full"`; per move `win_rate`, `score?`, `branches[]`, `ownership?: number[][]`; per branch `moves[]`, `win_rate`, `score?`, `visits` ("Number of times the AI considered the first move of this variation"), `lcb?`, `score_mean?`, `score_stdev?`, `utility?`, `utility_lcb?`, `policy?`.
- Board overlay (`src/components/AIReview/generateHeatmapAndMarks.ts::generateHeatmapAndMarks`): at most 6 branches (`branches.slice(0, 6)`) plus the played move if absent; the top branch is the "blue move" — solid circle `withAlpha(palette.blue_move, 0.7)`, border 0.2, colour `#0082FF` default; every other suggestion is a circle coloured by quality (score loss / win-rate loss, whichever is worse) with fill alpha `0.25 + 0.55 * (branch.visits / max_branch_visits)` ("Suggestions shade with their share of the visits, like the heatmap squares they replaced"); a subscript shows the delta (`formatDelta`), the blue move's "0" subscript is omitted; an optional second subscript shows visits formatted `10k`/`1.2k`/`836` (`formatVisits`, `show_visit_counts` preference `ai-review-show-visit-counts`).
- Quality colour scale: score-loss thresholds Excellent 0.2, Great 0.6, Good 1.2, Inaccuracy 4.0, Mistake 10.0 points (`goban/src/engine/ai/categorize.ts::DEFAULT_SCORE_DIFF_THRESHOLDS`); win-rate-loss thresholds 0.4/1.2/2.4/8.0/20.0 percentage points (`generateHeatmapAndMarks.ts::DEFAULT_WIN_RATE_DIFF_THRESHOLDS`); colours are read from CSS variables `--move-quality-*` / `--ai-blue-move` so board, badges and table agree (`resolveQualityPalette`); interpolation between band midpoints (`qualitySeverity`, `qualityColor`); "pure blue is reserved for the official blue move" (`EXCELLENT_BLEND_CAP = 0.75`).
- The old green heatmap primitive still exists in goban: one `<rect>` per cell, fill `#00FF00`, `fill-opacity = min(value, 0.5)`, only for values > 0.001 (`goban/src/Goban/SVGRenderer.ts::GCell.heatmap`, lines ~4939–4962); the AI review now calls `goban.setHeatmap(undefined)` and uses `setColoredCircles` instead (`AIReview.tsx` lines 476–478).
- Coloured circle primitive: one `<circle class="colored-circle">` per cell, radius `floor(square_size*0.5) - 0.5`, stroke width `radius * border_width` (`SVGRenderer.ts::GCell.circle`).
- Graph (`src/components/AIReview/ReviewChartD3.ts`): x = move number, y domain `[0,100]` in win-rate mode or `d3.extent` of scores in score mode (`updateScales`); area from the 50 % (or 0 score) midline with `curveMonotoneX`; the fill is "a flat two-tone split — solid dark where black is ahead ... solid light where white is ahead" via one gradient with a hard stop (`updateGradient` comment); a dashed variation line overlays the trunk line when browsing a variation; crosshairs for the current move, the variation move and the cursor; drag on the chart scrubs moves (`mouse_rect` handlers); highlighted worst moves are drawn as circles (`updateHighlightedMoves`); score/win-rate toggle persisted as preference `ai-review-use-score` (`AIReview.tsx` line 119; `ScoreWinRateToggle.tsx`).
- Ownership: `ownership?: number[][]` exists in the format (`JGOF.ts`), and the board pane can draw territory (`katrain` not OGS) — how OGS draws ownership was not traced (unverified).

### (c) Keyboard and touch
- Keyboard groups (`src/views/Game/game_keyboard_shortcuts.ts::GAME_KEYBOARD_SHORTCUT_GROUPS`): navigation `left`/`right` prev/next, `page-up`/`page-down` ±10 moves, `home`/`end` first/last, `up`/`down` previous/next variation branch, `space` play/pause autoplay; modes `shift-a` analyze, `shift-p` play mode, `shift-r` review, `shift-e` estimate score, `shift-i` "Show or hide the AI review", `shift-c` coordinates, `shift-z` zen mode, `escape` cancel/leave; variations `ctrl-c`/`ctrl-v`/`del` copy/paste/delete branch; analysis tools on F1–F8 ("Function keys must be enabled in Settings").
- Wheel over the board: `deltaY > 0` next move, `< 0` previous, gated by the `scroll_to_navigate` preference (`src/views/Game/Game.tsx::onWheel`).
- Touch on the board: goban binds document-level `touchstart/touchend/touchmove`; a tap counts only if the finger moved ≤ 10 px between start and end (`goban/src/Goban/SVGRenderer.ts` onTouchStart `activeTouchCompleter`, `Math.sqrt(...) <= 10`); mouse events are disabled for 5 s after any touch (`mouse_disabled = setTimeout(..., 5000)`); move submission on touch uses `one_click_submit` / `double_click_submit` (default true) (`goban/src/Goban/InteractiveBase.ts` lines 386–388). A helper toggles a `no-touch-action` class on the board container (`src/views/Game/touch_actions.ts`).
- Click handling is done on `mouseup` rather than `click` "since click events may not be synthesized reliably during rapid DOM updates (e.g., AI review streaming)" (`SVGRenderer.ts::onMouseUp` comment).

### (d) Absence and uncertainty
- Suggestions below `visits_threshold = Math.min(50, 0.1 * strength)` are dropped "unless none of the alternatives would qualify - then they are all shown. The blue move and the played move always show" (`generateHeatmapAndMarks.ts` lines 329–342, 356–360).
- The played move not among branches still gets a circle "colored and captioned by its positional delta when known", fill alpha 0.25, neutral grey `#888888` when no delta (`generateHeatmapAndMarks.ts` lines 445–487, `NEUTRAL_CIRCLE_COLOR`).
- Per-position fallbacks: if `moves[move_number]` is missing, win rate comes from `win_rates[move_number]` and score from the last known score walking backwards (`AIReview.tsx::updateHighlightsMarksAndHeatmaps` lines 358–376); for non-trunk nodes an analysis of the variation is requested on demand (`requestAnalysisOfVariation`).
- Chart: with zero entries a simplex-noise "animation wave" of 100 fake points is drawn and the plot is retried every 50 ms (`ReviewChartD3.ts::generateAnimationWave`, `prepareMainEntries`); for a `full` review still in progress, contiguous unanalysed ranges are drawn as `pending-backgrounds` behind the line, and "Fast reviews only analyze a subset of moves by design - they're not 'pending'" (`ReviewChartD3.ts::getPendingMoveRanges`).
- The whole board overlay can be switched off while keeping the panel (`showOnBoard`, preference `ai-review-show-on-board`; `AIReview.tsx` lines 397–402).

### (e) Deep-linking
- Routes `/game/:game_id/:move_number`, `/review/:review_id/:move_number`, `/demo/:review_id/:move_number` (`src/routes.tsx` lines 325–331); on load the page waits for `gamedata` / `review.load-end` then `gotoMove(parseInt(params.move_number))` (`src/views/Game/Game.tsx` lines 514–519). Share modal offers SGF/PNG/embed URLs but no move-number link (`src/views/Game/GameLinkModal.tsx`).

### (f) Performance at scale
- Review data arrives over a dedicated socket (`ai-review-connect`), and updates are coalesced with `requestAnimationFrame` before being applied and emitting targeted move/variation update events (`goban/src/engine/ai/AIReviewData.ts` lines 75–95, 191–247).
- Two review tiers, `fast` and `full`, with a "Full review" button when the current one is fast (`JGOF.ts::type`; `AIReview.tsx` lines 703–727).
- Mark writes are skipped if the engine has already navigated away ("writing would smear this move's marks onto another node") (`AIReview.tsx` lines 379–386).
- Portrait tab bar drops lower-priority tabs when they do not fit (`src/components/GobanView/util.ts::selectVisibleTabs`).

---

## 4. Sabaki (SabakiHQ/Sabaki + SabakiHQ/Shudan)

### (a) Layout
- Electron desktop app; the sidebar is a vertical `SplitContainer`: winrate graph on top (height `view.winrategraph_height`, only shown when `winrateData.some(x => x != null)`), then game tree + slider, then the comment box; sizes are user-draggable and persisted (`src/components/Sidebar.js` lines 125–235, `getDerivedStateFromProps`). No phone layout exists (desktop-only app; README).
- Main view: the board (`BoundedGoban`) with mode bars underneath (Play/Edit/Guess/Autoplay/Scoring/Find) (`src/components/MainView.js` lines 137–240).

### (b) Evaluation beside the position
- Engine protocol: `analyze <color> <interval>` streams `info move D4 visits 836 winrate 4656 prior 839 lcb 4640 order 0 pv D4 Q16 ...`; required keys `move`, `visits`, `winrate` (integer = percent × 100, float = 0–1), optional `scoreLead` (https://github.com/SabakiHQ/Sabaki/blob/master/docs/guides/engine-analysis-integration.md, read raw).
- Parser normalises winrate to percent and drops malformed segments (`src/modules/analysis.js::parseAnalysis`, `normalizeWinrate`).
- Heat map cell: `strength = round(visits * winrate * 8 / maxVisitsWin) + 1` (1–9; "the overlay color intensity always reflects the move's absolute quality"); label text = winrate (or scoreLead, or relative-to-best when `analysisValueType === 'relative'`) on line 1 and visits on line 2 (`1.2k` above 1000); text is empty when `visits < 10`; winrate is rounded to integer when `strength <= 3` else one decimal (`src/modules/analysis.js::getAnalysisHeatMapCell`; `src/components/Goban.js` lines 452–470).
- Shudan's `heatMap` prop takes `{strength: <integer>, text?: <string>}` per vertex and ships styles for strength 0–9; `paintMap` takes −1..1 (black/white paint) for ownership-style overlays; `ghostStoneMap` supports `'good' | 'interesting' | 'doubtful' | 'bad'` types; `lines` supports `'line'` and `'arrow'` (https://github.com/SabakiHQ/Shudan/blob/master/docs/README.md, read raw).
- Hovering a candidate replays its PV either instantly or move-by-move (`board.variation_replay_mode`, `board.variation_replay_interval`) (`src/components/Goban.js::playVariation`).
- Winrate graph: an SVG with `viewBox="0 0 ${width} 100"` where width = number of moves and y = percent (or score lead mapped to 0–100); a dashed 50 % guide line; a solid `#eee` polyline only where data exists (`M` restarts after a null), a dashed `#ccc` polyline bridging gaps; red blunder bars `M i,50 l 0,(50*x/dataDiffMax)` where `|diff| > blunderThreshold` (settings `view.winrategraph_blunderthreshold`, `_scorelead`); a strip shows current winrate and change with `+`/`−` and a `positive`/`negative` class beyond the threshold (`src/components/sidebars/WinrateGraph.js` lines 12–16, 30–65, 250–365).
- Analysis metric menu: Win Rate / Score Lead (`CmdOrCtrl+Shift+H` toggles), Absolute / Relative (`CmdOrCtrl+Shift+V`), Show Heatmap `CmdOrCtrl+H`, Show Analysis Graph `CmdOrCtrl+Shift+G`, Toggle Analysis `F4` (`src/menu.js` lines 498–499, 605–660, 765–780).

### (c) Keyboard and mouse
- Navigation accelerators: `Up` back, `Down` forward, `Home`/`End` beginning/end, `CmdOrCtrl+Up/Down` previous/next fork, `CmdOrCtrl+Shift+Up/Down` previous/next comment, `CmdOrCtrl+Left` main variation, `Left`/`Right` previous/next variation, `Shift+Left/Right` decrement/increment downstream variation, `CmdOrCtrl+G` go to move number, `CmdOrCtrl+PageUp/PageDown` previous/next game (`src/menu.js` lines 385–475).
- Holding `ArrowUp`/`ArrowDown` autoscrolls (`startAutoscrolling` on keydown, `stopAutoscrolling` on keyup) (`src/components/App.js` lines 143–156, 182–185).
- Wheel over the board, the game graph or the winrate graph steps moves once accumulated `deltaY` exceeds `game.navigation_sensitivity` (`src/components/App.js` lines 100–117).
- Clicking the winrate graph jumps to `round(width * percent)` (`WinrateGraph.js` lines 112–117).

### (d) Absence and uncertainty
- Per-move winrate is `null` at unanalysed nodes; the graph draws a gap (solid line restarts) and a dashed bridge; the readout shows `–` (`WinrateGraph.js` lines 48, 64, 258–260, 350–365).
- Board cells with `visits < 10` get colour but no text (`analysis.js::getAnalysisHeatMapCell`).
- The sidebar hides the whole graph when no node has data (`Sidebar.js` line 129).

### (e) Deep-linking — none (desktop app; `CmdOrCtrl+G` "Go to Move Number" is the equivalent, `src/menu.js` line 455).

### (f) Performance — no chunking/lazy-loading of analysis found; the graph's marker is guarded with `Number.isFinite` (`WinrateGraph.js` lines 400–405). Shudan exposes `BoundedGoban` (`maxWidth`/`maxHeight`/`maxVertexSize`, `onResized`) and `rangeX`/`rangeY` partial-board props (Shudan docs).

---

## 5. Lizzie (featurecat/lizzie)

The README documents only "hold down the key x" to see all commands (https://raw.githubusercontent.com/featurecat/lizzie/master/README.md, read raw); everything below is from source.

### (a) Layout
- Board is a square of side `maxSize = min(width, height)` centred vertically; left column holds captured stones, the winrate bar (height `maxSize/10`), the winrate graph (height `maxSize/3`), and the sub-board; the variation tree occupies the column right of the board (`src/main/java/featurecat/lizzie/gui/LizzieFrame.java` lines 326–370). No phone layout (Java desktop app).

### (b) Evaluation beside the position
- Candidate circles (`src/main/java/featurecat/lizzie/gui/BoardRenderer.java::drawLeelazSuggestionsBackgroundShadow`): each suggestion is a filled circle of stone radius; hue runs red→green by `percentPlayouts = playouts / maxPlayouts` (or by normalised winrate when `colorByWinrateInsteadOfVisits`), with a perceptual correction (`cbrt` below yellow, `sqrt` above); the best move (or max-winrate move in winrate mode) is cyan; alpha `minAlpha(20) + (maxAlpha(240) - 20) * max(0, log(percentPlayouts)/5 + 1)` so low-visit moves fade; moves with `playouts == 0` are skipped.
- Rings: best move ringed `Color.RED.brighter()` (stroke 2) when it is not also the max-winrate move; the max-winrate move ringed `Color.BLUE.brighter()`; both in one → darker circle colour (`BoardRenderer.java::drawLeelazSuggestionsBackgroundCircle`).
- Text: winrate `%.1f` (or handicap when `handicapInsteadOfWinrate`), playouts string, and optionally scoreMean; drawn only for the max-winrate move, moves with `percentPlayouts >= config.minPlayoutRatioForStats`, or the hovered move (`BoardRenderer.java::drawLeelazSuggestionsForeground`).
- Hovering a candidate shows its variation on the board (`branchOpt`), and `VK_UP/DOWN` or the wheel over a candidate step inside the branch (`Input.java` VK_UP/VK_DOWN `Lizzie.frame.doBranch(±1)`, `mouseWheelMoved`).
- Winrate graph (`src/main/java/featurecat/lizzie/gui/WinrateGraph.java::draw`): x = move number over 95 % of width, y = winrate from black's view, a dashed vertical line at the current move with the move number label; nodes with `playouts > 0` are plotted; line colour switches to `winrateMissLineColor` after a gap (`lastNodeOk`); optional blunder bars (`showBlunderBar`) whose height is a weighted function of the winrate change (`weightedBlunderBarHeight`: "<= 50% will use 75% of height, >= 50% will use 25% of height").

### (c) Keyboard and mouse
- (`src/main/java/featurecat/lizzie/gui/Input.java::keyPressed`) `Right` next branch / `Shift+Right` move branch down; `Left` previous branch / `Ctrl+Left` undo to first parent with variations; `Up` undo (`Ctrl` ×10, `Shift` to child of previous variation); `Down` redo (`Ctrl` ×10); `PageUp`/`PageDown` ±10 (`Ctrl+Shift` adjusts max alpha ±5); `Home` start (`Ctrl+Home` clear); `End` end; `Space` toggle ponder; `P` pass; `,` play best/current variation; `M` move numbers; `F` show next moves; `H` handicap-instead-of-winrate; `V` show branch; `W` toggle winrate (Ctrl: large winrate); `L` LCB winrate; `G` variation graph; `T` comments; `C` coordinates (Ctrl: copy SGF); `Ctrl+V` paste SGF; `Z`+Shift hints, `Alt+Z` sub-board; `A` toggle analysis (Ctrl clear, Alt avoid-move dialog); `B` show policy; `.` KataGo estimate modes; `D` cycle scoreMean/winrate text; `E` GTP console; `X` hold to show controls overlay; `0–9` switch engine.
- Wheel: forward = redo, backward = undo, or step the hovered branch (`Input.java::mouseWheelMoved`).

### (d) Absence and uncertainty
- Zero-playout suggestions are not drawn; low-playout text is suppressed by `minPlayoutRatioForStats`; alpha decays logarithmically with playout share (see (b)).
- Graph: nodes without playouts are skipped and the segment after a gap is drawn in `winrateMissLineColor` (`WinrateGraph.java` lines 165–176).

### (e) Deep-linking — none (desktop app).

### (f) Performance — the graph draws only up to `numMoves` (played moves) and the `x` overlay lists commands; nothing on chunking (unverified).

---

## 6. KaTrain (sanderland/katrain)

### (a) Layout
- Board with overlays on one side and "the right-hand side panels and options" whose state "is saved independently for 'play' and 'analyze'"; `~`/`` ` ``/`F12` cycle "more minimalistic UI modes" (README Analysis and Keyboard tables, read raw). Kivy desktop app; no phone layout documented.

### (b) Evaluation beside the position
- Instant feedback dots on played moves: "The colour indicates the size of the mistake according to KataGo. The size indicates if the mistake was actually punished. Going from fully punished at maximal size, to no actual effect on the score at minimal size" (README "Instant feedback"). Code: `evalscale = min(1, max(0, realized_points_lost / points_lost))`, dot radius `stone_size * (EVAL_DOT_MIN_SIZE 0.25 + evalscale * (0.5 - 0.25))` (`katrain/gui/badukpan.py` lines 293–296, 647–656; `katrain/gui/theme.py::EVAL_DOT_MAX_SIZE/MIN_SIZE`).
- Colour classes: `eval_thresholds = [12, 6, 3, 1.5, 0.5, 0]` points lost mapped by `evaluation_class` (first threshold the loss is ≥) to `EVAL_COLORS["theme:normal"]` = purple, red, orange, yellow, light green, green (`katrain/config.json` trainer; `katrain/core/utils.py::evaluation_class`; `theme.py::EVAL_COLORS`); a red-green-colourblind palette exists.
- `points_lost = player_sign * (parent_score - score)` between two analysed nodes (`katrain/core/game_node.py::points_lost`); candidate `pointsLost = player_sign(next_player) * (root_score - d["scoreLead"])`, `winrateLost` likewise (`game_node.py::candidate_moves`).
- Top moves (`e`): "Show the next moves KataGo considered, colored by their expected point loss. Small/faint dots indicate high uncertainty and never show text (lower than your 'fast visits' setting). Hover over any of them to see the principal variation, which is animated move by move" (README). Code: below `trainer/low_visits` (25) and not the engine's best and not a child → `scale = UNCERTAIN_HINT_SCALE 0.7`, `text_on = False`, `alpha = HINTS_LO_ALPHA 0.6`; otherwise `HINT_SCALE 0.98`, `HINTS_ALPHA 0.8`; text lines are `top_moves_show` (default delta score) over `top_moves_show_secondary` (default visits, `format_visits` → `1.2k`); the engine's `order == 0` move gets a cyan ring `TOP_MOVE_BORDER_COLOR` (`badukpan.py` lines 926–1027; `config.json` `top_moves_show`, `low_visits`).
- Policy (`r`): "where it thinks the best next move is purely from the position ... This turns off the 'top moves' setting" (README); drawn with `POLICY_ALPHA 0.5`, text only above a bound, best policy move marked (`badukpan.py` lines 709–760).
- Expected territory (`t`): "Show expected ownership of each intersection" (README); ownership alpha `abs(value) ** (1/OWNERSHIP_GAMMA 1.33)`, `TERRITORY_DISPLAY = "blended"` with `marks`/`blocks`/`shaded` alternatives, `BLOCKS_THRESHOLD 0.3` (`badukpan.py` lines 812–819; `theme.py` lines 131–140).
- Graph: `ScoreGraph` plots score (blue) and winrate (green) lines centred at zero with `score_scale` rounded up to a multiple of 5 and `winrate_scale` to a multiple of 10; missing values are `math.nan`; the highlight dot falls back to the last known value when the current node is NaN; clicking/dragging the graph navigates to the nearest node (`katrain/gui/widgets/graph.py::ScoreGraph.update_graph`, `on_touch_down/move/up`).

### (c) Keyboard and mouse
- README tables: `Tab` play/analyze; `q` child moves, `w` all dots, `e` top moves, `r` policy, `t` territory; `a` deeper, `s` equalize visits, `d` analyze all, `f` find alternatives, `g` area of interest, `Ctrl+h` reset, `i` insert mode, `l` play out, `Space` continuous analysis (`Shift+Space` without hints), `Enter` AI move, `F2` deeper full game, `F3` performance report, `F10` tsumego frame; `Alt` menu; `k` coordinates; `m` move numbers; `p` pass; `←`/`z` undo, `→`/`x` redo (`Shift` ×10, `Ctrl` to start/end); `↑/↓` branch; `Home/End`; `PageUp` make main branch; `Ctrl+Delete` delete node; `c` collapse; `b`/`Shift+b` back to branch point / main branch; `n`/`Shift+n` next/previous mistake; wheel over the right panel = redo/undo, over a candidate = scroll its PV; middle click adds the PV to the tree; click a move for statistics, double-click to navigate; drag-and-drop SGF; `Ctrl+v`/`Ctrl+c` clipboard; `Escape` stop analysis (README "Analysis" and "Keyboard and mouse shortcuts", read raw).

### (d) Absence and uncertainty
- Uncertain candidates: smaller (0.7×), fainter (0.6 alpha), no text (see (b)); "Equalize visits" exists precisely "to increase confidence in the suggestions with high uncertainty" (README `s`).
- `candidate_moves` with no per-move analysis falls back to the policy's top move with `pointsLost: 0` (`game_node.py::candidate_moves` lines 420–432); no analysis → empty list.
- Graph NaN for unanalysed nodes; ownership may be "one move out of date for smooth animation" (`badukpan.py` line 612 comment).

### (e) Deep-linking — none (desktop app).

### (f) Performance — two visit budgets `max_visits: 500` / `fast_visits: 25` (`config.json` engine); `Ctrl+v` loads with "a 'fast' analysis of the game With a high priority normal analysis for the last move" (README); analysis is stored packed (`game_node.py` `pack_floats`) — details not traced.

---

## 7. KataGo analysis engine output (lightvector/KataGo docs/Analysis_Engine.md, read raw)

Query fields relevant to a viewer: `analyzeTurns` ("0 is the initial position, 1 is the position after moves[0]"), `maxVisits`, `includeOwnership`, `includeOwnershipStdev`, `includeMovesOwnership`, `includePolicy`, `includePVVisits`, `reportDuringSearchEvery` ("report the partial analysis every that many seconds"), `priority`, `focusMoves`/`focusProb` (lines 86–120).

Response (lines 229–285):
- "All values will be from the perspective of `reportAnalysisWinratesAs` as specified in the analysis config file." (line 229)
- `isDuringSearch`: "Every position searched will still always conclude with exactly one final response ... where this field is false." (line 238); `turnNumber` (239).
- `moveInfos[]`: `move`; `visits` "The number of visits that the child node received"; `edgeVisits` "the number of visits that the root node 'wants' to invest in the move"; `winrate` "as a float in [0,1]"; `scoreMean` ("Same as scoreLead"); `scoreStdev` ("**significantly biased high** currently, although it can still be informative as *relative* indicator"); `scoreLead`; `scoreSelfplay`; `prior` "The policy prior of the move, as a float in [0,1]"; `utility` in `[-C,C]`; `lcb` "might lie outside of [0,1]"; `utilityLcb`; `weight`/`edgeWeight`; `order` "0 is the best"; `playSelectionValue` "The value used to compute order"; `isSymmetryOf`; `pv` "May be of variable length or even empty"; `pvVisits`/`pvEdgeVisits`; per-move `ownership`/`ownershipStdev` (lines 240–264).
- `rootInfo`: `winrate`, `scoreLead`, `scoreSelfplay`, `utility`, `visits`, plus `thisHash`, `symHash`, `currentPlayer`, `rawWinrate`, `rawLead`, `rawScoreSelfplay`, `rawScoreSelfplayStdev`, `rawStWrError`, `rawStScoreError`, `rawVarTimeLeft`, human-model fields; "properties of the root like 'winrate' and score will vary more smoothly and a bit more sluggishly than the corresponding property of the best move, since the rootInfo averages smoothly across all visits" (lines 265–282).
- `ownership`: "length boardYSize * boardXSize with values from -1 to 1 ... row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1)" (283); `ownershipStdev` 0–1 (284); `policy`: "length boardYSize * boardXSize + 1 with positive values summing to 1 ... and -1 indicating illegal moves ... The last value in the array is the policy value for passing" (285).
- Terminated queries "may be missing their data fields if no analysis at all was performed"; then only `id`, `turnNumber`, `isDuringSearch:false` and `noResults:true` are guaranteed (lines 349–351).
- This is the schema OGS's `JGOFAIReviewMoveVariation` (`visits`, `lcb`, `score_mean`, `score_stdev`, `utility`, `utility_lcb`, `policy`) and KaTrain's `candidate_moves` (`order`, `scoreLead`, `winrate`, `visits`, `pv`) consume.

---

## 8. TensorBoard scalars / Time Series dashboard (tensorflow/tensorboard)

### Layout and run model
- Docs: "Notice the 'Runs' selector on the left. A 'run' represents a set of logs from a round of training"; "Use the Runs selector to choose specific runs, or choose from only training or validation"; "Hover over the graph to see specific data points"; "TensorBoard has a smoothing parameter that you may need to turn down to zero to see the unsmoothed values" (https://www.tensorflow.org/tensorboard/scalars_and_keras, via fetch summary). "Scalars can be found in the Time Series or Scalars dashboards" (https://www.tensorflow.org/tensorboard/get_started, via fetch summary).
- Run colour: each run gets a colour id; the hex is `palette.colors[colorId % colors.length]`, light/dark variants, user override map wins, and runs without an id get the `inactive` grey (`tensorboard/webapp/util/ui_selectors.ts::getRunColorMap`). Default palette: Slate `#425066`/`#8e98a3`, Cyan `#12b5cb`, Pink `#e52592`, Yellow `#f9ab00`, Purple `#9334e6`, Light Green `#7cb342`, Orange `#e8710a`; inactive Gray `#e0e0e0`/`#3b3b3b` (`tensorboard/webapp/util/colors.ts::DEFAULT_PALETTE`). README: "click on the colored circles in the run selector to change a run's color" (https://github.com/tensorflow/tensorboard/blob/master/README.md, via fetch summary).
- Grouping keys: `GroupByKey.RUN`, `EXPERIMENT`, `REGEX` (`tensorboard/webapp/runs/types.ts::GroupByKey`).
- Settings defaults: `tooltipSort: NEAREST`, `ignoreOutliers: true`, `xAxisType: STEP`, `hideEmptyCards: true`, `scalarSmoothing: 0.6`, `scalarPartitionNonMonotonicX: false` (`tensorboard/webapp/metrics/store/metrics_types.ts::METRICS_SETTINGS_DEFAULT`).

### Smoothing (exact)
- `classicSmoothing(data, w)`: `w` clamped to [0,1]; constant series returned unchanged ("See #786"); for each finite `y`: `last = last * w + (1 - w) * y; numAccum++; debiasWeight = w === 1 ? 1 : 1 - w^numAccum; out = last / debiasWeight`; non-finite points pass through unchanged and do not advance `numAccum` (`tensorboard/webapp/widgets/line_chart_v2/data_transformer.ts::classicSmoothing`). The in-source comment gives the bias derivation `EMA = c*(1 - s^t)`.
- Known distortion (from the code): the EMA lags — it is a one-sided filter with no time weighting, so after a step change the smoothed line trails by ~`1/(1-w)` samples; it is applied after downsampling, so its effective window depends on how many points survived reservoir sampling (inference from `classicSmoothing` + reservoir; the lag magnitude is not stated in source — unverified as a documented claim).
- UI: slider max `0.99`, numeric input max `SCALARS_SMOOTHING_MAX = 0.999`, because "When smoothing === 1, all lines become flat on the x-axis, which is not useful at all" (`tensorboard/webapp/metrics/views/right_pane/settings_view_component.ts` lines 44–56; `tensorboard/webapp/metrics/internal_types.ts::SCALARS_SMOOTHING_MAX`).

### X-axis
- `XAxisType.STEP = 'step'`, `RELATIVE = 'relative'`, `WALL_TIME = 'walltime'` (`tensorboard/webapp/metrics/internal_types.ts::XAxisType`); dropdown labels "Step", "Relative", "Wall" under "Horizontal Axis" (`settings_view_component.ts::XAxisTypeDropdownOptions`; `settings_view_component.ng.html` line 21). Relative x is `point.wallTime - firstWallTime` of that series (ms); wall time is `stepDatum.wallTime * 1000` (`tensorboard/webapp/metrics/views/card_renderer/scalar_card_container.ts` lines 405–431, 688–696). Step selector / linked time is "Only available when Horizontal Axis is set to step" (`settings_view_component.ng.html` line 32).

### "Ignore outliers in chart scaling"
- Checkbox label verbatim (`settings_view_component.ng.html` line 153). Implementation: collect finite y of visible, non-aux series, sort ascending, and if enabled and n > 2 take `yMin = y[ceil((n-1)*0.05)]`, `yMax = y[floor((n-1)*0.95)]` — the 5th/95th percentiles of all points across visible runs (`tensorboard/webapp/widgets/line_chart_v2/line_chart_internal_utils.ts::computeDataSeriesExtent`).

### Tooltips
- Sort options: Alphabetical, Ascending, Descending, Nearest Pixel, Nearest Y (`settings_view_component.ts::TooltipSortDropdownOptions`); optional "Limit tooltip rows to" N (`settings_view_component.ng.html` lines 157–169).

### Downsampling
- README: "TensorBoard uses reservoir sampling to downsample your data so that it can be loaded into RAM. You can modify the number of elements it will keep per tag by using the `--samples_per_plugin` command line argument (ex: `--samples_per_plugin=scalars=500,images=20`)" (README via fetch summary). CLI help: defaults "10 for images, 500 for histograms, and 1000 for scalars" (`tensorboard/plugins/core/core_plugin.py` `--samples_per_plugin` help text, lines 640–653).
- Algorithm: deterministic seed (`random.Random(0)`); once full, `r = randint(0, num_items_seen)`; if `r < max_size` pop item `r` and append the new one, else if `always_keep_last` replace the last item; "With probability (_max_size/_num_items_seen) a random item in the bucket will be popped out and the new item will be appended to the end" (`tensorboard/backend/event_processing/reservoir.py::_ReservoirBucket.AddItem`, docstring and lines 219–229). Note the last item is always the newest but the sample is otherwise random, so the kept set is not evenly spaced.

### NaN / missing
- Smoothing passes non-finite y through (`classicSmoothing`). Extent ignores non-finite via `isSafeNumber = Number.isFinite(x)` (`tensorboard/webapp/widgets/line_chart_v2/lib/scale.ts` lines 96–97; log scale additionally requires `x > 0`, line 180).
- Rendering: the polyline is partitioned into NUMBER and NAN runs; a NUMBER run of a single point is drawn as a circle (radius 4); NaN points are substituted with the last legal point's coordinates (or the data origin) and drawn as triangles of size 12 in the series colour (skipped for aux series) (`tensorboard/webapp/widgets/line_chart_v2/lib/series_line_view.ts::partitionPolyline`, `recordPartition`, `redraw`).

### Deep-linking / performance
- Not researched beyond the above; the Time Series settings persist (`persistent_settings`) and `hideEmptyCards` is on by default (`metrics_types.ts`). Card width slider 335–735 px (`settings_view_component.ts` lines 41–42).

---

## 9. Hex-grid rendering — Red Blob Games "Hexagonal Grids" (https://www.redblobgames.com/grids/hexagons/, raw HTML read; https://www.redblobgames.com/grids/hexagons/implementation.html, via fetch summary)

- Orientation: "flat top angles are 0°, 60°, 120°, 180°, 240°, 300° and pointy top angles are 30°, 90°, 150°, 210°, 270°, 330°"; corner `i` is at `angle_deg = 60 * i` (+30 for pointy), `Point(center.x + size*cos, center.y + size*sin)`; "the diagrams on this page use the y axis pointing down (angles increase clockwise)" (Angles section).
- Spacing: pointy top "horiz = width == sqrt(3) * size == 2 * inradius", "vert = 3/4 * height == 3/2 * size"; flat top "horiz = 3/4 * width = 3/2 * size", "vert = height = sqrt(3) * size = 2 * inradius" (Spacing section).
- Coordinates: cube `q + r + s = 0`; axial "is the same as the cube system except we don't store the s coordinate"; doubled has constraint `(col + row) mod 2 == 0`; "I like cube coordinates for algorithms and axial or doubled for storage" (Coordinate Systems section).
- Hex→pixel (axial): pointy `x = size * (sqrt(3)*q + sqrt(3)/2*r)`, `y = size * (3/2 * r)`; flat `x = size * (3/2 * q)`, `y = size * (sqrt(3)/2*q + sqrt(3)*r)`; presented as matrix `[[sqrt(3), sqrt(3)/2],[0, 3/2]]` and `[[3/2, 0],[sqrt(3)/2, sqrt(3)]]` times `size` (Hex to pixel section, `pointy_hex_to_pixel`, `flat_hex_to_pixel`). Origin offset: add `origin.x/y` at the end and subtract it first when inverting (Mod: non-zero origin).
- Pixel→hex: divide by size, then pointy `q = sqrt(3)/3*x - 1/3*y`, `r = 2/3*y`; flat `q = 2/3*x`, `r = -1/3*x + sqrt(3)/3*y`; then `axial_round` (Pixel to Hex section, `pixel_to_pointy_hex`, `pixel_to_flat_hex`).
- Rounding: `cube_round` rounds q,r,s, computes diffs, and resets the component with the largest diff so `q + r + s = 0`; `axial_round(hex) = cube_to_axial(cube_round(axial_to_cube(hex)))` (Rounding section, verbatim algorithm).
- Neighbours: six axial direction vectors `[+1,0] [+1,-1] [0,-1] [-1,0] [-1,+1] [0,+1]`; distance `max(|dq|,|dr|,|ds|)` = `(|dq|+|dr|+|ds|)/2` (Neighbors/Distances, via fetch summary of the same page).
- Range: all hexes within N: `for -N ≤ q ≤ N: for max(-N,-q-N) ≤ r ≤ min(N,-q+N)` (Range section). Line drawing: lerp then `cube_round` with N = distance (Line drawing section).
- Storage: "Use a 2D Array ... Store Hex(q, r) at array[r][q]"; "Use a hash table instead of dense array. This allows arbitrarily shaped maps, including ones with holes. Store Hex(q, r) in hash_table(hash(q, r))"; rectangle: "Store Hex(q, r) at array[r][q + floor(r/2)]" (Map storage section).
- Implementation page: `Orientation` = forward matrix `f0..f3`, inverse `b0..b3`, `start_angle`; pointy `[√3, √3/2, 0, 3/2]`, `[√3/3, −1/3, 0, 2/3]`, 0.5; flat `[3/2, 0, √3/2, √3]`, `[2/3, 0, −1/3, √3/3]`, 0.0; `Layout(orientation, size, origin)`; corner offset `angle = 2π(start_angle + corner)/6` (implementation.html, via fetch summary).
- SVG: the page says only "This page includes interactive diagrams that require your browser to have SVG and Javascript enabled" and that "The interactive diagrams are in diagrams.js and index.js, using Vue.js to inject into the templates"; it gives no `<use>`/`<defs>` or performance guidance for drawing many hexes (absence verified in the page text).

---

## Brief: Weights & Biases and Plotly / Vega-Lite (primary docs, via fetch summary)

- W&B: "Each run is given a random color by default upon initialization"; colours are changeable "through the run table or the chart legend settings"; line plots offer x-axis "training steps (default), relative time, and wall time" plus custom logged x (https://docs.wandb.ai/guides/app/features/panels/line-plot/). Smoothing offers time-weighted EMA (default, "takes the density of points on the line ... into account"), Gaussian, running average and EMA, and for the EMAs "A debias term is added so that early values in the time series aren't biased towards zero" (https://docs.wandb.ai/guides/app/features/panels/line-plot/smoothing). Comparing runs: pinned/baseline runs with "summary metric deltas show how each run compares to the baseline"; hide with the eye icon; "if runs are grouped by a column, pinned and baseline runs are not visually distinct" (https://docs.wandb.ai/guides/runs/compare-runs). Group charting options (mean/min-max/std-dev bands) were not found on the grouping page fetched (https://docs.wandb.ai/guides/runs/grouping) — unverified.
- Vega-Lite: "A Trellis plot (or small multiple) is a series of similar plots that displays different subsets of the same data, facilitating comparison across subsets"; channels `facet`, `row`, `column`; `columns` wraps (undefined = single row; 1 = vconcat); "The default resolutions for row/column facet are shared scales, axes, and legends", override with `resolve: "independent"`; `spacing` default 20 px; headers label rows/columns (https://vega.github.io/vega-lite/docs/facet.html). Plotly: "Facet plots, also known as trellis plots or small multiples, are figures made up of multiple subplots which have the same set of axes"; `facet_row`, `facet_col`, `facet_col_wrap`; axes are linked by default ("zooming in one facet also zooms the others"), decoupled with `update_yaxes(matches=None)`; reduce `facet_row_spacing`/`facet_col_spacing` toward 0.01 for many facets (https://plotly.com/python/facet-plots/).

---

## Takes and rejects — DECIDED for the observatory

The rule that decides most rows is the house's own: **absent is not zero** (R333(d) clause 3,
R352(g) clause 3), **every number traces to a contract field**, **extremes exact, nothing
smoothed** (the dashboard's per-pixel envelope), and **no producer, no panel** (LAW-07 / R4).
Where a reference does something better than the two existing tools, it is taken; where it
draws what it does not know, it is rejected by name.

### lichess
- TAKE the one-grid contract whose areas re-flow between phone and desk (`ui/analyse/css/_layout.scss`): the position page is one grid — desk `board | side`, phone `board / controls / readout / turns` — pure CSS, no script, offline-safe. The canvas's desk and phone position artboards are that grid.
- TAKE "the number moves into the controls bar on a phone" (`controls.ts::renderMobileCevalTab`): the root value and the tactical state ride the readout under the board on the phone; no gauge column exists at 390 px.
- TAKE the per-move chart conventions — a bounded symmetric quantity, a fixed ±1 axis, the current ply marked, click-to-jump (`ui/chart/src/acpl.ts`): the root value is already in [−1, 1]; the value trace on the position page is that chart.
- TAKE `#<ply>`-style addressing with debounced `replaceState` (`ctrl.ts::updateHref`): the observatory keeps its address in the query (`?ply=N&heat=1&tac=1`) so the server can render that ply without script; the debounce is taken as is.
- TAKE hold-to-repeat on prev/next and `touch-action: pan-y` on the controls (`controls.ts::holdControl`).
- REJECT the cp→win-chance sigmoid and the mate vocabulary (`winningChances.ts`): the record carries a root value, not centipawns.
- REJECT arrows sized by chance loss (`autoShape.ts`): a compound-turn hex game has no single move to point at; per-child visits on cells are the encoding that survives.
- REJECT holding the last gauge value at 50 % opacity when there is no eval (`renderGauge` `gaugeLast`): a stale number drawn as current is the F-10 class in pixels; a ply without a root draws nothing and says so.

### chess.com
- TAKE "provisional until the full review runs" as a rendered state (the classification-change article): the R353(b) PROVISIONAL badge on every sealbot reading is exactly this, and it is on the canvas.
- TAKE the arrows at both ends of the list and a "next key move" button (the app article) in spirit: `[` `]` step rounds, `n` `p` step games, and the round page links the next flagged game.
- REJECT move-classification badges with withheld thresholds and rating dependence (Brilliant … Blunder; CAPS2 "the math … has changed"): the observatory shows the exact one-turn fact (CHECK / LOST1 / missed block) with its derivation named, never a grade.
- REJECT swipe-on-the-list instead of the board (the app article) — partly: the existing viewer's board swipe (40 px threshold) is kept because a tap-to-cell is distinguishable from a swipe by distance (OGS uses the same 10 px tap rule); the turn list scrolls, it does not scrub.
- REJECT depth tiers behind a paywall: every position gets the root the run recorded; there is no deeper-on-demand.

### OGS
- TAKE the candidate encoding's two channels — the argmax child ringed and uniquely marked, the others' fill alpha `0.25 + 0.55 × visits/max` with the count as text (`generateHeatmapAndMarks.ts`): the board's visits overlay is one hue (the sequential blue) at that alpha rule with the count printed, the played cell ringed; the palette is CSS custom properties so light and dark agree.
- TAKE the visit floor with its fallback — hide children under a floor unless nothing qualifies (`generateHeatmapAndMarks.ts` lines 329–342): at 64-sim fast-arm roots most children carry 1–3 visits; the floor is stated on the panel and never blanks the board.
- TAKE pending ranges drawn as gaps with a background and the fast/full distinction (`ReviewChartD3.ts::getPendingMoveRanges`): plies with no root (the opponent's, or self-play) are gaps on the value trace, labelled, never interpolated; a channel that records no roots by design is a stated gap, not "pending".
- TAKE the route-segment address `/game/:id/:move_number` (`routes.tsx`) as `/run/<id>/game/<game_id>?ply=N`.
- TAKE the portrait rule — one scroll column, board sized to keep the readout on screen, panels below (`GobanView.css`), decided by aspect ratio and a 1280 px cap (`util.ts::goban_view_mode`), rather than a width breakpoint alone.
- REJECT the two-metric loss colour scale (score AND win-rate loss, worst wins): one value, one scale.
- REJECT the simplex-noise "animation wave" on an empty chart (`ReviewChartD3.ts::generateAnimationWave`): it draws fake data where there is none — the exact thing the house rule forbids; a hatched panel with a sentence instead.
- REJECT document-level touch listeners with a 5 s mouse blackout (`SVGRenderer.ts` `mouse_disabled`): Pointer Events on the board's own element.

### Sabaki / Shudan
- TAKE the winrate-graph gap grammar — a solid line only where data exists, a dashed bridge across `null` spans, `–` in the readout (`WinrateGraph.js`): the value trace's absence grammar, verbatim.
- TAKE `viewBox="0 0 <moves> 100"` with `vector-effect: non-scaling-stroke` (`WinrateGraph.js`): one unit per ply, scales to any width, one file.
- TAKE the heat-cell contract `{strength 0–9, text?}` with text suppressed under a visit floor (`analysis.js::getAnalysisHeatMapCell`, Shudan docs) as the shape of the server-rendered board's cell classes.
- REJECT `strength = visits × winrate / max` (`analysis.js`): it conflates two quantities; visits and value stay on separate channels (the trace carries value, the cells carry visits).
- REJECT Electron accelerators (`CmdOrCtrl+…`, `src/menu.js`) as the model: single keys that do not fight the browser.
- Hover-to-replay a PV (`Goban.js::playVariation`) is NOT taken: the record stores no principal variation (contract #11 carries `visits` only) — a stated gap, not a feature.

### Lizzie
- TAKE the log-alpha fade for low-playout children and "text only above a share or on hover" (`BoardRenderer.java::drawLeelazSuggestionsBackgroundShadow`, `drawLeelazSuggestionsForeground`) as the fallback for crowded roots (the frontier's 512-sim roots have 8–20 visited children).
- TAKE the dashed current-move line with the ply printed beside it on the graph (`WinrateGraph.java`).
- REJECT red→green hue by visits with cyan for the best (`BoardRenderer.java`): colour-blind-hostile, conflates "most searched" with "good", breaks the scale; one hue for visits, a ring for the argmax (the dataviz palette method's sequential rule).
- REJECT the two-ring semantics (best by visits vs best by value): one root value, one argmax.
- REJECT hold-`x` for help: a `?` overlay (lichess) and the keys printed on the Tokens sheet.

### KaTrain
- TAKE the uncertain-hint rule — under a visit floor draw 0.7× size, 0.6 alpha, no text (`badukpan.py` lines 941–949): the three-way downgrade for low-visit children on the phone.
- TAKE explicit thresholds in config plus a colour-blind palette (`config.json` `eval_thresholds`, `theme.py::EVAL_COLORS`) in spirit: every threshold the observatory draws is on the record (`monitor_gates.ply_cap_abort_rate`) or stated as the minted term it reads at; the palette is validated for CVD by the dataviz script rather than offered as an alternate.
- TAKE the graph's NaN discipline and click/drag-to-navigate (`graph.py::ScoreGraph`).
- REJECT the two-channel mistake dot (colour = size of the mistake, size = how much was realised): it needs a per-move engine judgement the record does not carry; the record's one exact fact per turn (missed block / missed win) is a badge, not a dot.
- REJECT policy and ownership overlays (`r`, `t`): no producer on the record (LAW-07); the keys are not reserved either — a key that does nothing is a lie in the key map.

### KataGo analysis engine
- TAKE the minimal root schema — `rootInfo{visits, winrate}` + `moveInfos[]{move, visits, order}` with the perspective stated once (`Analysis_Engine.md` lines 229–282) — as the shape the position JSON takes: `root_value` with `by` (whose search) already IS the perspective statement in contract #11; the JSON adds `order` and the argmax, nothing else.
- TAKE the warning that root aggregates lag the best child (line 282): the trace plots the root value; the argmax child's visits are its own label.
- TAKE `noResults: true` (lines 349–351) as the shape of "no root at this ply": a named absence field, never zeros.
- REJECT `scoreLead` / `scoreStdev` / `utility`: no score in a connection game, and the doc calls `scoreStdev` "significantly biased high".

### TensorBoard
- TAKE the run-colour rule: a colour is a property of the run id, stable across every panel, with light/dark pairs (`ui_selectors.ts::getRunColorMap`, `colors.ts`) — the dataviz "colour follows the entity" rule, applied to runs on the compare page.
- TAKE the NaN rendering — gaps split the polyline, a single surviving point is a circle, a hole is marked (`series_line_view.ts`): the envelope charts already skip non-finite values; a gap is drawn as a gap.
- TAKE the x-axis triple Step / Relative / Wall (`scalar_card_container.ts`) as an OPTION on compare only, default step — relative time is how two runs' throughput are honestly compared (the eval rounds land at different steps).
- REJECT the debiased EMA (`data_transformer.ts::classicSmoothing`) and the 0.6 default: the house draws the per-pixel envelope so the extremes are exact; a smoothing lever hides the spike the trainer's reader is looking for (R351(d)'s trough, R352(c)'s cap window are both spikes).
- REJECT "ignore outliers in chart scaling" (`line_chart_internal_utils.ts::computeDataSeriesExtent`, the 5th–95th percentiles): a clipped axis hides a firing; the compare page instead sets the axis from the drawn lines and prints the min/max of every row in the caption.
- REJECT reservoir sampling for a record (`reservoir.py::AddItem`): seeded-random, only the last point guaranteed, peaks vanish, two renders differ; the envelope's per-pixel min–max–last is deterministic and keeps every extreme.
- REJECT hiding empty cards (`hideEmptyCards: true`): a missing producer is a stated gap on the page, never a hidden card.

### Red Blob Games
- TAKE axial `(q, r)`, pointy-top, `x = size(√3·q + √3/2·r)`, `y = size·3/2·r`, the inverse and `cube_round` for tap-to-cell (Hex to pixel / Pixel to Hex / Rounding): the viewer already uses the forward map; the inverse lands in `board.js` for the phone's tap-to-ply.
- TAKE the hash-table storage for unbounded maps (Map storage): stones keyed by `(q, r)`, the drawn window from the bounding box plus a margin (as today).
- TAKE the range loop for the empty-cell ring around the played area.
- REJECT offset/doubled coordinates anywhere in state: the record is axial (contract #11) and stays so.
- The page carries no SVG performance guidance (verified absence); one `<path>` for the empty grid and one element per stone is what the current viewer does at 256-ply games without trouble and is kept until measured otherwise.

### W&B, Vega-Lite, Plotly (small multiples)
- TAKE shared scales and axes across facets by default (Vega-Lite `resolve`, Plotly's linked facet axes) for the compare page's small multiples — one x (step), each multiple its own y.
- TAKE W&B's "pinned baseline with deltas" in spirit: the compare page's instrument table states the difference between the two runs' instruments before any curve is read.
- REJECT W&B's random run colours: fixed slot by run order, stated in the legend.

## URLs fetched

- https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md — reachable; field reference (fetch summary), then raw file read: https://raw.githubusercontent.com/lightvector/KataGo/master/docs/Analysis_Engine.md — reachable, 560 lines.
- https://www.redblobgames.com/grids/hexagons/ — reachable; fetch summary plus raw HTML (186 KB) grepped for formulas/spacing/storage/SVG text.
- https://www.redblobgames.com/grids/hexagons/implementation.html — reachable; orientation matrices, layout, rounding (fetch summary).
- https://github.com/featurecat/lizzie/blob/master/README.md and the raw README — reachable; only the "hold x" hint.
- https://raw.githubusercontent.com/featurecat/lizzie/master/src/main/java/featurecat/lizzie/gui/{BoardRenderer,Input,WinrateGraph,LizzieFrame,VariationTree}.java — reachable; candidate drawing, keys, graph, layout.
- https://github.com/sanderland/katrain/blob/master/README.md and raw — reachable; analysis/keyboard tables.
- https://raw.githubusercontent.com/sanderland/katrain/master/katrain/{gui/badukpan.py,gui/theme.py,gui/widgets/graph.py,core/game_node.py,core/utils.py,core/constants.py,config.json} — reachable.
- https://github.com/SabakiHQ/Sabaki/blob/master/README.md, docs/README.md, docs/guides/engine-analysis-integration.md (raw) — reachable.
- https://raw.githubusercontent.com/SabakiHQ/Sabaki/master/src/{components/Goban.js,components/sidebars/WinrateGraph.js,components/Sidebar.js,components/MainView.js,components/App.js,menu.js,modules/analysis.js} — reachable.
- https://raw.githubusercontent.com/SabakiHQ/Shudan/master/README.md and docs/README.md — reachable; prop docs.
- https://github.com/tensorflow/tensorboard/blob/master/README.md — reachable (fetch summary).
- https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/{webapp/widgets/line_chart_v2/data_transformer.ts, webapp/widgets/line_chart_v2/line_chart_internal_utils.ts, webapp/widgets/line_chart_v2/lib/series_line_view.ts, webapp/widgets/line_chart_v2/lib/scale.ts, webapp/widgets/line_chart_v2/lib/renderer/svg_renderer.ts, backend/event_processing/reservoir.py, webapp/metrics/store/metrics_types.ts, webapp/metrics/internal_types.ts, webapp/metrics/views/right_pane/settings_view_component.ts, webapp/metrics/views/right_pane/settings_view_component.ng.html, webapp/metrics/views/card_renderer/scalar_card_container.ts, webapp/util/colors.ts, webapp/util/ui_selectors.ts, webapp/runs/types.ts, plugins/core/core_plugin.py} — reachable.
- https://www.tensorflow.org/tensorboard/scalars_and_keras and /get_started — reachable (fetch summary).
- https://raw.githubusercontent.com/lichess-org/lila/master/ui/analyse/src/{keyboard.ts,ctrl.ts,autoShape.ts,serverSideUnderboard.ts,evalCache.ts,view/main.ts,view/components.ts,view/controls.ts}, ui/analyse/css/{_layout.scss,_tools-mobile.scss}, ui/lib/src/ceval/{winningChances.ts,view/main.ts,util.ts,types.ts}, ui/lib/css/ceval/_eval-gauge.scss, ui/chart/src/{acpl.ts,index.ts,interface.ts,chart.game.ts} — reachable. (ui/ceval/src/… paths returned 404: the module moved to ui/lib/src/ceval.)
- https://lichess.org/analysis/help — reachable; keyboard help overlay text (fetch summary).
- https://api.github.com/repos/{lichess-org/lila,online-go/online-go.com,online-go/goban,SabakiHQ/Sabaki,SabakiHQ/Shudan,featurecat/lizzie,tensorflow/tensorboard}/contents/… — directory listings; reachable. GitHub code search via `gh api search/code` for tensorboard symbols — reachable.
- https://support.chess.com/en/collections/13175943-analysis-game-review — reachable; article index.
- https://support.chess.com/en/articles/{8572705,8584089,10328363,8648715,8583825,8708970,8708936,9462780} — reachable (fetch summaries).
- https://raw.githubusercontent.com/online-go/online-go.com/master/src/{components/AIReview/AIReview.tsx,components/AIReview/generateHeatmapAndMarks.ts,components/AIReview/ReviewChartD3.ts,components/AIReview/ReviewChart.tsx,components/AIReview/hooks.ts,components/AIReview/utils.ts,components/AIReview/AIReview.css,components/GobanView/GobanView.tsx,components/GobanView/GobanView.css,components/GobanView/util.ts,views/Game/Game.tsx,views/Game/Game.css,views/Game/game_keyboard_shortcuts.ts,views/Game/touch_actions.ts,views/Game/GameLinkModal.tsx,routes.tsx} — reachable. (src/lib/ai_review_utils.ts — 404; the constants live in goban.)
- https://raw.githubusercontent.com/online-go/goban/main/src/{Goban/SVGRenderer.ts,Goban/InteractiveBase.ts,Goban/README.md,engine/formats/JGOF.ts,engine/ai/categorize.ts,engine/ai/AIReviewData.ts} — reachable.
- https://docs.wandb.ai/guides/runs/compare-runs, /guides/runs/grouping, /guides/app/features/panels/line-plot/, /guides/app/features/panels/line-plot/smoothing — reachable (fetch summaries).
- https://vega.github.io/vega-lite/docs/facet.html — reachable (fetch summary).
- https://plotly.com/python/facet-plots/ — reachable (fetch summary).
