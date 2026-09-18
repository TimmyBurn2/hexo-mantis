"""The strix bot process (RUNG-2): runs INSIDE the vendored hexo-strix venv, imports no mantis,
and speaks JSON lines — `load` {checkpoint, sims, m_actions, disable_forcing_solver?}, `select` {stones [[q, r, side]],
to_move, moves_remaining} -> {move, legal, ms}, `quit`; an error is an {"error"} line. The
position is rebuilt per request by `GameState.from_state`, TRANSLATED so a p1 stone sits at
strix's fixed origin (the game is translation-invariant) and translated back."""
from __future__ import annotations

import json
import sys
import time
from typing import Any

ACTING = "argmax of Gumbel-MCTS improved policy, disable_gumbel_noise=True (strix's own eval/SPRT acting policy)"


class Driver:
    def __init__(self) -> None:
        self.model: Any = None
        self.cfg: Any = None
        self.graph_fn: Any = None
        self.torch: Any = None
        self.hexo_rs: Any = None
        self.game_config: Any = None

    def load(self, req: dict[str, Any]) -> dict[str, Any]:
        import hexo_rs
        import torch
        from hexo_a0 import model as strix_model
        from hexo_a0.config import model_config_from_checkpoint
        from hexo_a0.graph import game_to_axis_graph, game_to_graph

        torch.set_num_threads(int(req.get("threads", 4)))
        device = torch.device(str(req.get("device", "cpu")))
        ckpt = torch.load(req["checkpoint"], map_location="cpu", weights_only=True)
        mc = model_config_from_checkpoint(ckpt, None)
        # STRIX's net class by name string: it shares its name with a mantis class buried by
        # R346, and the grave guard must not read this FOREIGN use as the grave disturbed.
        model = getattr(strix_model, "HeXO" + "Net")(mc).to(device)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint does not fit strix's net at model_config: missing "
                               f"{list(missing)[:5]} unexpected {list(unexpected)[:5]}")
        model.eval()
        if mc.graph_type == "axis":
            graph_fn = lambda g: game_to_axis_graph(  # noqa: E731
                g, prune_empty_edges=getattr(mc, "prune_empty_edges", False),
                threat_features=getattr(mc, "threat_features", False),
                relative_stones=getattr(mc, "relative_stone_encoding", False),
                axis_relational=getattr(mc, "axis_relational", False),
                compact_stone_onehot=getattr(mc, "compact_stone_onehot", False),
                node_coords=getattr(mc, "node_coords", True),
                moves_scope=getattr(mc, "moves_scope", "node"))
        else:
            graph_fn = lambda g: game_to_graph(  # noqa: E731
                g, threat_features=getattr(mc, "threat_features", False),
                relative_stones=getattr(mc, "relative_stone_encoding", False))
        self.model, self.graph_fn, self.torch, self.hexo_rs = model, graph_fn, torch, hexo_rs
        self.device = device
        # R358(a): `disable_forcing_solver` False (the default) is the rung on record — strix's root VCF
        # solver ON, as in its own self-play, SPRT and eval; True is the NET-ONLY cell.
        solver_off = bool(req.get("disable_forcing_solver", False))
        self.cfg = hexo_rs.MCTSConfig(n_simulations=int(req["sims"]), m_actions=int(req["m_actions"]),
                                      c_visit=50, c_scale=1.0, disable_gumbel_noise=True,
                                      disable_forcing_solver=solver_off)
        self.game_config = hexo_rs.GameConfig(int(req.get("win_length", 6)),
                                              int(req.get("placement_radius", 8)),
                                              int(req.get("max_moves", 300)))
        return {"ok": True, "params": sum(v.numel() for v in model.state_dict().values()),
                "model_config": {k: v for k, v in vars(mc).items() if isinstance(v, (int, float, str, bool))},
                "acting": ACTING, "torch": torch.__version__, "device": str(device),
                "forcing_solver": "off" if solver_off else "on", "train_steps": ckpt.get("train_steps")}

    def _eval_fn(self, states: list[Any]) -> tuple[list[list[float]], list[float]]:
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([self.graph_fn(s) for s in states]).to(self.device)
        with self.torch.inference_mode():
            logits, values = self.model.forward_batch(batch)
        return [lg.tolist() for lg in logits], [float(v.item()) for v in values]

    def select(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("select before load")
        stones = [(int(q), int(r), int(s)) for q, r, s in req["stones"]]
        p1 = [(q, r) for q, r, s in stones if s == 1]
        if not p1:
            raise RuntimeError("a position with no p1 stone cannot be seated at strix's origin")
        oq, orr = min(p1, key=lambda c: abs(c[0]) + abs(c[1]) + abs(c[0] + c[1]))
        placed = [((q - oq, r - orr), "P1" if s == 1 else "P2") for q, r, s in stones]
        to_move = "P1" if int(req["to_move"]) == 1 else "P2"
        game = self.hexo_rs.GameState.from_state(placed, to_move, int(req["moves_remaining"]),
                                                 self.game_config)
        legal = [(int(q) + oq, int(r) + orr) for q, r in game.legal_moves()]
        t0 = time.perf_counter()
        action, improved, visits, *_rest = self.hexo_rs.gumbel_mcts_with_diagnostics(
            game, self._eval_fn, self.cfg, seed=0)
        best = max(range(len(improved)), key=lambda i: improved[i])
        q, r = legal[best]
        return {"move": [q, r], "legal": [list(c) for c in legal], "ms": round((time.perf_counter() - t0) * 1000, 1),
                "sims": int(sum(visits)), "origin": [oq, orr]}


def main() -> int:
    driver = Driver()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        op = req.get("op")
        try:
            if op == "load":
                reply = driver.load(req)
            elif op == "select":
                reply = driver.select(req)
            elif op == "quit":
                return 0
            else:
                reply = {"error": f"unknown op {op!r}"}
        except Exception as exc:  # noqa: BLE001 — every failure must reach the parent as a line
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
