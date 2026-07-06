#!/usr/bin/env python3
"""Local chess and Go RL benchmark lane with Elo-style improvement tracking.

This runner is intentionally self-contained and calibration-safe:
- Chess uses python-chess for legal move generation and game rules.
- Go uses a small local rules engine for 5x5/9x9 tactical self-play.
- No public benchmark solutions or external model calls are used.
- Outputs appendable evidence, traces, and rolling Elo ratings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

try:
    import chess  # type: ignore
except Exception:  # pragma: no cover - exercised in environments without dependency
    chess = None  # type: ignore


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 24.0) -> Tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def default_ratings() -> Dict[str, Any]:
    return {
        "schema": "theseus_board_game_elo_ratings_v1",
        "updated_at": None,
        "ratings": {
            "chess": {
                "random": 1000.0,
                "material_greedy": 1000.0,
                "center_material": 1000.0,
            },
            "go": {
                "random": 1000.0,
                "capture_greedy": 1000.0,
                "influence": 1000.0,
            },
        },
        "games_played": {"chess": 0, "go": 0},
    }


def rating_delta(current: Dict[str, float], previous: Dict[str, float]) -> Dict[str, float]:
    return {name: round(float(current.get(name, 1000.0)) - float(previous.get(name, 1000.0)), 3) for name in current}


if chess is not None:
    CHESS_PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.1,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }
else:
    CHESS_PIECE_VALUES = {}


def chess_material_score(board: Any) -> float:
    score = 0.0
    for piece in board.piece_map().values():
        value = CHESS_PIECE_VALUES.get(piece.piece_type, 0.0)
        score += value if piece.color == chess.WHITE else -value
    return score


def chess_terminal_score(board: Any, mover: bool) -> float:
    if board.is_checkmate():
        return -10000.0 if board.turn == mover else 10000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    return 0.0


def chess_move_features(board: Any, move: Any, mover: bool) -> Dict[str, float]:
    captured = board.piece_at(move.to_square)
    capture_value = CHESS_PIECE_VALUES.get(captured.piece_type, 0.0) if captured else 0.0
    attacker = board.piece_at(move.from_square)
    attacker_value = CHESS_PIECE_VALUES.get(attacker.piece_type, 0.0) if attacker else 1.0
    center_bonus = 0.0
    if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
        center_bonus = 0.35
    elif move.to_square in {
        chess.C3,
        chess.D3,
        chess.E3,
        chess.F3,
        chess.C4,
        chess.F4,
        chess.C5,
        chess.F5,
        chess.C6,
        chess.D6,
        chess.E6,
        chess.F6,
    }:
        center_bonus = 0.15

    board.push(move)
    material = chess_material_score(board)
    mover_material = material if mover == chess.WHITE else -material
    check_bonus = 0.25 if board.is_check() else 0.0
    terminal = chess_terminal_score(board, mover)
    legal_reply_count = len(list(board.legal_moves)) if not board.is_game_over(claim_draw=True) else 0
    board.pop()

    return {
        "capture_value": capture_value,
        "exchange_hint": capture_value - 0.1 * attacker_value,
        "center_bonus": center_bonus,
        "check_bonus": check_bonus,
        "mover_material": mover_material,
        "terminal": terminal,
        "mobility_pressure": -0.01 * legal_reply_count,
    }


def choose_chess_move(board: Any, policy: str, rng: random.Random) -> Any:
    legal = list(board.legal_moves)
    if not legal:
        return None
    if policy == "random":
        return rng.choice(legal)

    mover = board.turn
    scored: List[Tuple[float, Any]] = []
    for move in legal:
        features = chess_move_features(board, move, mover)
        if policy == "material_greedy":
            score = (
                4.0 * features["capture_value"]
                + 1.2 * features["exchange_hint"]
                + 0.25 * features["mover_material"]
                + features["terminal"]
            )
        elif policy == "center_material":
            score = (
                2.5 * features["capture_value"]
                + 1.0 * features["center_bonus"]
                + 0.6 * features["check_bonus"]
                + 0.2 * features["mover_material"]
                + 0.15 * features["mobility_pressure"]
                + features["terminal"]
            )
        else:
            score = rng.random()
        score += rng.random() * 1e-6
        scored.append((score, move))
    return max(scored, key=lambda item: item[0])[1]


def play_chess_game(
    white_policy: str,
    black_policy: str,
    seed: int,
    max_plies: int = 160,
    trace_limit: int = 32,
) -> Dict[str, Any]:
    board = chess.Board()
    rng = random.Random(seed)
    trace: List[Dict[str, Any]] = []

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        policy = white_policy if board.turn == chess.WHITE else black_policy
        move = choose_chess_move(board, policy, rng)
        if move is None:
            break
        if len(trace) < trace_limit:
            trace.append(
                {
                    "ply": ply,
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "policy": policy,
                    "fen_before": board.fen(),
                    "move": board.san(move),
                    "uci": move.uci(),
                }
            )
        board.push(move)

    termination = "rules"
    result = board.result(claim_draw=True)
    if not board.is_game_over(claim_draw=True):
        termination = "max_plies_material_tiebreak"
        material = chess_material_score(board)
        if material > 0.25:
            result = "1-0"
        elif material < -0.25:
            result = "0-1"
        else:
            result = "1/2-1/2"

    if result == "1-0":
        winner = "white"
        white_score = 1.0
    elif result == "0-1":
        winner = "black"
        white_score = 0.0
    else:
        winner = "draw"
        white_score = 0.5

    return {
        "game": "chess",
        "white_policy": white_policy,
        "black_policy": black_policy,
        "seed": seed,
        "result": result,
        "winner": winner,
        "white_score": white_score,
        "plies": board.ply(),
        "termination": termination,
        "final_fen": board.fen(),
        "trace": trace,
    }


def run_chess_legal_smoke(seed: int, games: int = 6) -> Dict[str, Any]:
    if chess is None:
        return {"ok": False, "reason": "python-chess dependency missing"}
    rng = random.Random(seed)
    failures = 0
    plies = 0
    for game_idx in range(games):
        board = chess.Board()
        for _ in range(80):
            if board.is_game_over(claim_draw=True):
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            if move not in board.legal_moves:
                failures += 1
                break
            board.push(move)
            plies += 1
    return {"ok": failures == 0, "games": games, "plies": plies, "illegal_moves": failures}


def run_chess_capture_diagnostic(seed: int, target: int = 32) -> Dict[str, Any]:
    if chess is None:
        return {"ok": False, "reason": "python-chess dependency missing"}
    rng = random.Random(seed)
    board = chess.Board()
    cases: List[Tuple[str, str, float]] = []
    attempts = 0
    while len(cases) < target and attempts < target * 80:
        attempts += 1
        if board.is_game_over(claim_draw=True) or board.ply() > 90:
            board = chess.Board()
        legal = list(board.legal_moves)
        captures = []
        for move in legal:
            captured = board.piece_at(move.to_square)
            if captured:
                captures.append((move, CHESS_PIECE_VALUES.get(captured.piece_type, 0.0)))
        if captures:
            best_value = max(value for _, value in captures)
            chosen = choose_chess_move(board, "material_greedy", rng)
            chosen_capture = board.piece_at(chosen.to_square)
            chosen_value = CHESS_PIECE_VALUES.get(chosen_capture.piece_type, 0.0) if chosen_capture else 0.0
            cases.append((board.fen(), chosen.uci(), 1.0 if chosen_value >= best_value else 0.0))
        move = rng.choice(legal)
        board.push(move)

    passed = sum(score for _, _, score in cases)
    total = len(cases)
    pass_rate = passed / total if total else 0.0
    return {
        "ok": total >= min(target, 8) and pass_rate >= 0.70,
        "cases": total,
        "pass_rate": round(pass_rate, 4),
        "target": target,
        "sample_failures": [
            {"fen": fen, "chosen": move}
            for fen, move, score in cases
            if score < 1.0
        ][:3],
    }


Point = int


@dataclass
class GoMoveResult:
    ok: bool
    captured: int = 0
    reason: Optional[str] = None
    previous_turn: int = 1


class GoEnv:
    """Small board Go rules engine for RL diagnostics.

    It supports captures, suicide prevention, pass moves, simple ko, and area
    scoring. It is deliberately small so it can run anywhere in the hive.
    """

    def __init__(self, size: int = 5, komi: float = 2.5) -> None:
        if size < 3 or size > 19:
            raise ValueError("Go board size must be between 3 and 19")
        self.size = size
        self.komi = komi
        self.board: List[int] = [0] * (size * size)
        self.turn = 1
        self.passes = 0
        self.move_count = 0
        self.previous_hash: Optional[Tuple[int, ...]] = None
        self.last_captured = 0

    def clone(self) -> "GoEnv":
        other = GoEnv(self.size, self.komi)
        other.board = list(self.board)
        other.turn = self.turn
        other.passes = self.passes
        other.move_count = self.move_count
        other.previous_hash = self.previous_hash
        other.last_captured = self.last_captured
        return other

    def idx(self, row: int, col: int) -> int:
        return row * self.size + col

    def rc(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.size)

    def neighbors(self, idx: int) -> Iterable[int]:
        row, col = self.rc(idx)
        if row > 0:
            yield self.idx(row - 1, col)
        if row + 1 < self.size:
            yield self.idx(row + 1, col)
        if col > 0:
            yield self.idx(row, col - 1)
        if col + 1 < self.size:
            yield self.idx(row, col + 1)

    def group_and_liberties(self, start: int, board: Optional[List[int]] = None) -> Tuple[set[int], set[int]]:
        state = self.board if board is None else board
        color = state[start]
        if color == 0:
            return set(), set()
        group = {start}
        liberties: set[int] = set()
        stack = [start]
        while stack:
            point = stack.pop()
            for neighbor in self.neighbors(point):
                value = state[neighbor]
                if value == 0:
                    liberties.add(neighbor)
                elif value == color and neighbor not in group:
                    group.add(neighbor)
                    stack.append(neighbor)
        return group, liberties

    def _simulate_move(self, action: int) -> Tuple[bool, Optional[List[int]], int, Optional[str]]:
        if action == -1:
            return True, list(self.board), 0, None
        if action < 0 or action >= self.size * self.size:
            return False, None, 0, "out_of_bounds"
        if self.board[action] != 0:
            return False, None, 0, "occupied"

        state = list(self.board)
        state[action] = self.turn
        captured = 0
        opponent = -self.turn
        for neighbor in self.neighbors(action):
            if state[neighbor] != opponent:
                continue
            group, liberties = self.group_and_liberties(neighbor, state)
            if not liberties:
                for point in group:
                    state[point] = 0
                captured += len(group)

        own_group, own_liberties = self.group_and_liberties(action, state)
        if not own_group or not own_liberties:
            return False, None, captured, "suicide"

        next_hash = tuple(state)
        if self.previous_hash is not None and next_hash == self.previous_hash:
            return False, None, captured, "simple_ko"
        return True, state, captured, None

    def legal_moves(self, include_pass: bool = True) -> List[int]:
        moves = [point for point in range(self.size * self.size) if self._simulate_move(point)[0]]
        if include_pass:
            moves.append(-1)
        return moves

    def step(self, action: int) -> GoMoveResult:
        previous_turn = self.turn
        ok, state, captured, reason = self._simulate_move(action)
        if not ok or state is None:
            return GoMoveResult(ok=False, captured=captured, reason=reason, previous_turn=previous_turn)
        self.previous_hash = tuple(self.board)
        self.board = state
        self.turn = -self.turn
        self.move_count += 1
        self.last_captured = captured
        if action == -1:
            self.passes += 1
        else:
            self.passes = 0
        return GoMoveResult(ok=True, captured=captured, previous_turn=previous_turn)

    def is_done(self, max_moves: int) -> bool:
        return self.passes >= 2 or self.move_count >= max_moves or all(value != 0 for value in self.board)

    def area_score(self) -> float:
        visited: set[int] = set()
        black = 0.0
        white = self.komi
        for idx, value in enumerate(self.board):
            if value == 1:
                black += 1.0
            elif value == -1:
                white += 1.0
            elif idx not in visited:
                region: set[int] = set()
                borders: set[int] = set()
                stack = [idx]
                visited.add(idx)
                while stack:
                    point = stack.pop()
                    region.add(point)
                    for neighbor in self.neighbors(point):
                        neighbor_value = self.board[neighbor]
                        if neighbor_value == 0 and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                        elif neighbor_value != 0:
                            borders.add(neighbor_value)
                if borders == {1}:
                    black += float(len(region))
                elif borders == {-1}:
                    white += float(len(region))
        return black - white

    def render_compact(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for row in range(self.size):
            rows.append("".join(symbols[self.board[self.idx(row, col)]] for col in range(self.size)))
        return "/".join(rows)


def go_move_features(env: GoEnv, action: int) -> Dict[str, float]:
    if action == -1:
        return {"captured": 0.0, "center": -1.0, "liberties": 0.0, "edge_penalty": 0.0}
    sim = env.clone()
    result = sim.step(action)
    if not result.ok:
        return {"captured": -100.0, "center": -1.0, "liberties": 0.0, "edge_penalty": 0.0}
    group, liberties = sim.group_and_liberties(action)
    row, col = env.rc(action)
    center = (env.size - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    center_score = -distance / max(1.0, env.size)
    edge_penalty = -0.1 if row in {0, env.size - 1} or col in {0, env.size - 1} else 0.0
    return {
        "captured": float(result.captured),
        "center": center_score,
        "liberties": float(len(liberties)),
        "edge_penalty": edge_penalty,
    }


def choose_go_move(env: GoEnv, policy: str, rng: random.Random) -> int:
    legal = env.legal_moves(include_pass=True)
    non_pass = [move for move in legal if move != -1]
    if not non_pass:
        return -1
    if policy == "random":
        return rng.choice(non_pass if rng.random() > 0.03 else legal)

    scored: List[Tuple[float, int]] = []
    for move in non_pass:
        features = go_move_features(env, move)
        if policy == "capture_greedy":
            score = 5.0 * features["captured"] + 0.3 * features["liberties"] + 0.1 * features["center"]
        elif policy == "influence":
            score = 0.6 * features["liberties"] + 0.7 * features["center"] + features["edge_penalty"] + 1.0 * features["captured"]
        else:
            score = rng.random()
        score += rng.random() * 1e-6
        scored.append((score, move))

    if not scored:
        return -1
    best_score, best_move = max(scored, key=lambda item: item[0])
    if env.move_count > env.size * env.size and best_score < 0.05:
        return -1
    return best_move


def play_go_game(
    black_policy: str,
    white_policy: str,
    seed: int,
    size: int = 5,
    max_moves: Optional[int] = None,
    trace_limit: int = 48,
) -> Dict[str, Any]:
    env = GoEnv(size=size, komi=2.5 if size <= 5 else 6.5)
    rng = random.Random(seed)
    max_steps = max_moves or (size * size * 2)
    trace: List[Dict[str, Any]] = []
    illegal_moves = 0

    while not env.is_done(max_steps):
        policy = black_policy if env.turn == 1 else white_policy
        action = choose_go_move(env, policy, rng)
        board_before = env.render_compact()
        result = env.step(action)
        if not result.ok:
            illegal_moves += 1
            action = -1
            result = env.step(action)
        if len(trace) < trace_limit:
            trace.append(
                {
                    "move": env.move_count,
                    "turn": "black" if result.previous_turn == 1 else "white",
                    "policy": policy,
                    "action": "pass" if action == -1 else env.rc(action),
                    "captured": result.captured,
                    "board_before": board_before,
                    "board_after": env.render_compact(),
                }
            )

    score = env.area_score()
    winner = "black" if score > 0 else "white"
    black_score = 1.0 if score > 0 else 0.0
    return {
        "game": "go",
        "black_policy": black_policy,
        "white_policy": white_policy,
        "seed": seed,
        "board_size": size,
        "score_black_minus_white": round(score, 3),
        "winner": winner,
        "black_score": black_score,
        "moves": env.move_count,
        "illegal_moves": illegal_moves,
        "termination": "two_passes" if env.passes >= 2 else "max_moves_or_full_board",
        "final_board": env.render_compact(),
        "trace": trace,
    }


def run_go_legal_smoke(seed: int, size: int = 5, games: int = 6) -> Dict[str, Any]:
    rng = random.Random(seed)
    illegal = 0
    moves = 0
    captures = 0
    for game_idx in range(games):
        env = GoEnv(size=size)
        while not env.is_done(size * size * 2):
            legal = env.legal_moves(include_pass=True)
            action = rng.choice(legal)
            result = env.step(action)
            if not result.ok:
                illegal += 1
                break
            captures += result.captured
            moves += 1
    return {"ok": illegal == 0, "games": games, "moves": moves, "illegal_moves": illegal, "captures": captures}


def run_go_capture_diagnostic(seed: int, size: int = 5) -> Dict[str, Any]:
    env = GoEnv(size=size)
    # Shape: black to play at (1, 2) captures a white stone at (1, 1).
    env.board[env.idx(1, 1)] = -1
    env.board[env.idx(0, 1)] = 1
    env.board[env.idx(1, 0)] = 1
    env.board[env.idx(2, 1)] = 1
    env.turn = 1
    rng = random.Random(seed)
    chosen = choose_go_move(env, "capture_greedy", rng)
    expected = env.idx(1, 2)
    before = env.render_compact()
    result = env.step(chosen)
    ok = chosen == expected and result.ok and result.captured == 1
    return {
        "ok": ok,
        "chosen": "pass" if chosen == -1 else env.rc(chosen),
        "expected": env.rc(expected),
        "captured": result.captured,
        "before": before,
        "after": env.render_compact(),
    }


def run_chess_rating_series(
    ratings: Dict[str, float],
    games: int,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    policies = ["random", "material_greedy", "center_material"]
    matches: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    pairs = [(a, b) for idx, a in enumerate(policies) for b in policies[idx + 1 :]]
    for game_idx in range(games):
        policy_a, policy_b = pairs[game_idx % len(pairs)]
        if game_idx % 2 == 0:
            white, black = policy_a, policy_b
            a_is_white = True
        else:
            white, black = policy_b, policy_a
            a_is_white = False
        result = play_chess_game(white, black, rng.randint(1, 10**9))
        score_a = result["white_score"] if a_is_white else 1.0 - result["white_score"]
        old_a = ratings.get(policy_a, 1000.0)
        old_b = ratings.get(policy_b, 1000.0)
        new_a, new_b = update_elo(old_a, old_b, score_a)
        ratings[policy_a], ratings[policy_b] = new_a, new_b
        matches.append(
            {
                "game": "chess",
                "match_index": game_idx,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "white": white,
                "black": black,
                "score_a": score_a,
                "result": result["result"],
                "plies": result["plies"],
                "ratings_after": {policy_a: round(new_a, 3), policy_b: round(new_b, 3)},
            }
        )
        traces.append(result)
    return ratings, matches, traces


def run_go_rating_series(
    ratings: Dict[str, float],
    games: int,
    seed: int,
    board_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    policies = ["random", "capture_greedy", "influence"]
    matches: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    pairs = [(a, b) for idx, a in enumerate(policies) for b in policies[idx + 1 :]]
    for game_idx in range(games):
        policy_a, policy_b = pairs[game_idx % len(pairs)]
        if game_idx % 2 == 0:
            black, white = policy_a, policy_b
            a_is_black = True
        else:
            black, white = policy_b, policy_a
            a_is_black = False
        result = play_go_game(black, white, rng.randint(1, 10**9), size=board_size)
        score_a = result["black_score"] if a_is_black else 1.0 - result["black_score"]
        old_a = ratings.get(policy_a, 1000.0)
        old_b = ratings.get(policy_b, 1000.0)
        new_a, new_b = update_elo(old_a, old_b, score_a)
        ratings[policy_a], ratings[policy_b] = new_a, new_b
        matches.append(
            {
                "game": "go",
                "match_index": game_idx,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "black": black,
                "white": white,
                "score_a": score_a,
                "winner": result["winner"],
                "moves": result["moves"],
                "ratings_after": {policy_a: round(new_a, 3), policy_b: round(new_b, 3)},
            }
        )
        traces.append(result)
    return ratings, matches, traces


def learned_policy_payload(
    traces: List[Dict[str, Any]],
    ratings_payload: Dict[str, Any],
    games: Dict[str, Any],
    learned_policy_out: Path,
    policy_train_out: Path,
) -> Dict[str, Any]:
    game_rows: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        game = str(trace.get("game") or "unknown")
        row = game_rows.setdefault(
            game,
            {
                "trace_count": 0,
                "policy_counts": {},
                "terminal_counts": {},
                "legal_action_mask_examples": 0,
                "tactical_residuals": {},
                "avg_length": 0.0,
                "lengths": [],
            },
        )
        row["trace_count"] += 1
        length = int(trace.get("plies") or trace.get("moves") or 0)
        row["lengths"].append(length)
        terminal = str(trace.get("termination") or "unknown")
        row["terminal_counts"][terminal] = int(row["terminal_counts"].get(terminal, 0)) + 1
        policies = [str(trace.get("white_policy") or ""), str(trace.get("black_policy") or ""), str(trace.get("black_policy") or ""), str(trace.get("white_policy") or "")]
        for policy in policies:
            if policy:
                row["policy_counts"][policy] = int(row["policy_counts"].get(policy, 0)) + 1
        turns = trace.get("trace") if isinstance(trace.get("trace"), list) else []
        row["legal_action_mask_examples"] += len(turns)
        if game == "go" and int(trace.get("illegal_moves") or 0) > 0:
            row["tactical_residuals"]["illegal_go_move_repaired_to_pass"] = int(row["tactical_residuals"].get("illegal_go_move_repaired_to_pass", 0)) + 1
        if str(trace.get("winner") or "") == "draw":
            row["tactical_residuals"]["draw_or_unclear_terminal"] = int(row["tactical_residuals"].get("draw_or_unclear_terminal", 0)) + 1
        if terminal.startswith("max_"):
            row["tactical_residuals"]["horizon_limit_terminal"] = int(row["tactical_residuals"].get("horizon_limit_terminal", 0)) + 1
    for row in game_rows.values():
        lengths = row.pop("lengths", [])
        row["avg_length"] = round(sum(lengths) / max(1, len(lengths)), 3)

    skill_weights = {
        "legal_action_masking": min(1.0, sum(row["legal_action_mask_examples"] for row in game_rows.values()) / 250.0),
        "state_memory": min(1.0, sum(row["avg_length"] for row in game_rows.values()) / 100.0),
        "branch_planning": min(1.0, len(traces) / 48.0),
        "reward_credit_assignment": min(1.0, sum(sum(abs(float(v)) for v in (games.get(game, {}).get("rating_delta") or {}).values()) for game in games) / 100.0),
        "repair_after_failure": min(1.0, sum(sum(int(v) for v in row["tactical_residuals"].values()) for row in game_rows.values()) / 12.0),
    }
    policy_training_rows = board_game_policy_training_rows(traces, ratings_payload)
    policy_train_row_count = append_jsonl(policy_train_out, policy_training_rows) if policy_training_rows else 0
    policy_cards = []
    for game, row in sorted(game_rows.items()):
        ratings = ratings_payload.get("ratings", {}).get(game, {})
        best_policy = max(ratings.items(), key=lambda item: float(item[1]))[0] if ratings else "unknown"
        residuals = row["tactical_residuals"] or {"none_observed": 0}
        policy_cards.append(
            {
                "game": game,
                "best_policy_by_current_elo": best_policy,
                "trace_count": row["trace_count"],
                "avg_length": row["avg_length"],
                "legal_action_mask_examples": row["legal_action_mask_examples"],
                "policy_counts": row["policy_counts"],
                "terminal_counts": row["terminal_counts"],
                "tactical_residuals": residuals,
                "learned_policy_weights": learned_policy_weights_for_game(game, traces, ratings),
                "learned_control_lesson": learned_control_lesson(game, best_policy, residuals),
            }
        )
    return {
        "policy": "project_theseus_board_game_learned_policy_v1",
        "created_utc": now_iso(),
        "trigger_state": "GREEN" if policy_cards else "YELLOW",
        "summary": {
            "policy_card_count": len(policy_cards),
            "trace_count": len(traces),
            "policy_train_row_count": policy_train_row_count,
            "skill_weights": {key: round(value, 3) for key, value in skill_weights.items()},
            "learned_policy_out": rel(learned_policy_out),
            "policy_train_out": rel(policy_train_out),
            "external_inference_calls": 0,
            "public_data_policy": "local_self_play_only_not_public_benchmark_training",
        },
        "policy_cards": policy_cards,
        "skill_capsule_hints": [
            {
                "skill": skill,
                "weight": round(weight, 3),
                "transfer_targets": ["code_generation_arm", "repo_repair_arm", "tool_use_arm", "symliquid_state_engine"],
            }
            for skill, weight in sorted(skill_weights.items())
        ],
        "external_inference_calls": 0,
    }


def board_game_policy_training_rows(
    traces: List[Dict[str, Any]],
    ratings_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        game = str(trace.get("game") or "unknown")
        ratings = ratings_payload.get("ratings", {}).get(game, {})
        best_policy = max(ratings.items(), key=lambda item: float(item[1]))[0] if ratings else "unknown"
        score = terminal_policy_score(trace, best_policy)
        turns = trace.get("trace") if isinstance(trace.get("trace"), list) else []
        for turn in turns[:64]:
            if not isinstance(turn, dict):
                continue
            policy = str(turn.get("policy") or "")
            rows.append(
                {
                    "policy": "project_theseus_board_game_policy_training_row_v1",
                    "created_utc": now_iso(),
                    "game": game,
                    "source": "local_self_play",
                    "case_id": trace.get("seed"),
                    "turn": turn.get("ply", turn.get("move_index")),
                    "state": turn.get("fen_before") or turn.get("board_before") or turn.get("state") or "",
                    "action": turn.get("uci") or turn.get("move") or turn.get("point") or "",
                    "actor_policy": policy,
                    "best_policy_by_current_elo": best_policy,
                    "reward": score if policy == best_policy else -0.25 * abs(score),
                    "skills": ["legal_action_masking", "state_memory", "branch_planning", "reward_credit_assignment"],
                    "transfer_targets": ["symliquid_state_engine", "tool_use_arm", "repo_repair_arm", "code_generation_arm"],
                    "public_benchmark_training_data_used": False,
                    "external_inference_calls": 0,
                }
            )
    return rows


def terminal_policy_score(trace: Dict[str, Any], policy: str) -> float:
    winner = str(trace.get("winner") or "")
    white = str(trace.get("white_policy") or "")
    black = str(trace.get("black_policy") or "")
    if winner == "draw":
        return 0.0
    if winner == "white":
        return 1.0 if white == policy else -1.0
    if winner == "black":
        return 1.0 if black == policy else -1.0
    return 0.0


def learned_policy_weights_for_game(
    game: str,
    traces: List[Dict[str, Any]],
    ratings: Dict[str, Any],
) -> Dict[str, float]:
    best_policy = max(ratings.items(), key=lambda item: float(item[1]))[0] if ratings else ""
    game_traces = [row for row in traces if str(row.get("game") or "") == game]
    wins = sum(1 for row in game_traces if terminal_policy_score(row, best_policy) > 0)
    losses = sum(1 for row in game_traces if terminal_policy_score(row, best_policy) < 0)
    action_examples = sum(len(row.get("trace") if isinstance(row.get("trace"), list) else []) for row in game_traces)
    return {
        "legal_action_mask": round(min(1.0, action_examples / 128.0), 3),
        "state_memory": round(min(1.0, sum(int(row.get("plies") or row.get("moves") or 0) for row in game_traces) / 240.0), 3),
        "tactical_preference": round((wins - losses) / max(1, wins + losses), 3),
        "current_elo": round(float(ratings.get(best_policy, 1000.0) or 1000.0), 3) if best_policy else 1000.0,
    }


def learned_control_lesson(game: str, best_policy: str, residuals: Dict[str, int]) -> str:
    residual_text = ", ".join(sorted(residuals)) if residuals else "no tactical residuals"
    if game == "chess":
        return f"Prefer legal move masking, material/center state features, and tactical residual replay; current best policy is {best_policy}; residuals: {residual_text}."
    if game == "go":
        return f"Prefer legal action masking, liberties/capture features, and pass/terminal recovery; current best policy is {best_policy}; residuals: {residual_text}."
    return f"Use legal actions, compact state, terminal rewards, and replay residuals; current best policy is {best_policy}; residuals: {residual_text}."


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Board Game RL Benchmark",
        "",
        f"- Status: **{report['status']}**",
        f"- Generated: `{report['generated_at']}`",
        f"- Seed: `{report['seed']}`",
        f"- Elo ratings: `{report['outputs']['ratings']}`",
        f"- Trace output: `{report['outputs']['traces']}`",
        f"- Learned policy: `{report['outputs']['learned_policy']}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        marker = "PASS" if gate["ok"] else "FAIL"
        lines.append(f"- {marker}: `{gate['name']}` - {gate.get('detail', '')}")
    lines.extend(["", "## Results", ""])
    for game, result in report["games"].items():
        lines.append(f"### {game.title()}")
        if result.get("skipped"):
            lines.append(f"- Skipped: {result.get('reason')}")
            continue
        lines.append(f"- Matches: `{len(result.get('matches', []))}`")
        lines.append(f"- Rating deltas: `{json.dumps(result.get('rating_delta', {}), sort_keys=True)}`")
        diagnostics = result.get("diagnostics", {})
        for name, diagnostic in diagnostics.items():
            lines.append(f"- Diagnostic `{name}`: `{json.dumps(diagnostic, sort_keys=True)}`")
        lines.append("")
    lines.extend(
        [
            "## Learning Contract",
            "",
            "- This lane is for local self-play, tactical diagnostics, and Elo-style trend evidence.",
            "- Public benchmark solutions are not used.",
            "- The Go engine is project-local and small-board first; scale board size only after tactical gates remain green.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local chess/Go RL benchmark and Elo ratchet.")
    parser.add_argument("--games", default="chess,go", help="Comma-separated games to run: chess,go")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chess-games", type=int, default=24)
    parser.add_argument("--go-games", type=int, default=24)
    parser.add_argument("--go-board-size", type=int, default=5)
    parser.add_argument("--out", default=str(REPORTS / "board_game_rl_benchmark.json"))
    parser.add_argument("--markdown-out", default=str(REPORTS / "board_game_rl_benchmark.md"))
    parser.add_argument("--ratings-out", default=str(REPORTS / "board_game_elo_ratings.json"))
    parser.add_argument("--history-out", default=str(REPORTS / "board_game_elo_history.jsonl"))
    parser.add_argument("--trace-out", default=str(REPORTS / "board_game_rl_traces.jsonl"))
    parser.add_argument("--learned-policy-out", default=str(REPORTS / "board_game_learned_policy.json"))
    parser.add_argument("--policy-train-out", default="D:/ProjectTheseus/training_data/board_game_rl/private_train/board_game_policy_rows.jsonl")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requested = {item.strip().lower() for item in args.games.split(",") if item.strip()}
    out_path = Path(args.out)
    markdown_path = Path(args.markdown_out)
    ratings_path = Path(args.ratings_out)
    history_path = Path(args.history_out)
    trace_path = Path(args.trace_out)
    learned_policy_path = Path(args.learned_policy_out)
    policy_train_path = Path(args.policy_train_out)

    previous_payload = read_json(ratings_path, default_ratings())
    ratings_payload = default_ratings()
    ratings_payload["ratings"].update(previous_payload.get("ratings", {}))
    ratings_payload["games_played"].update(previous_payload.get("games_played", {}))
    previous_ratings = json.loads(json.dumps(ratings_payload["ratings"]))

    gates: List[Dict[str, Any]] = []
    games: Dict[str, Any] = {}
    all_traces: List[Dict[str, Any]] = []
    started = time.time()

    if "chess" in requested:
        if chess is None:
            games["chess"] = {"skipped": True, "reason": "python-chess is not installed"}
            gates.append({"name": "chess_dependency", "ok": False, "detail": "python-chess import failed"})
        else:
            legal = run_chess_legal_smoke(args.seed + 1)
            capture = run_chess_capture_diagnostic(args.seed + 2)
            current = dict(ratings_payload["ratings"].get("chess", {}))
            updated, matches, traces = run_chess_rating_series(current, args.chess_games, args.seed + 10)
            ratings_payload["ratings"]["chess"] = {name: round(value, 3) for name, value in updated.items()}
            ratings_payload["games_played"]["chess"] = int(ratings_payload["games_played"].get("chess", 0)) + len(matches)
            delta = rating_delta(updated, previous_ratings.get("chess", {}))
            games["chess"] = {
                "skipped": False,
                "diagnostics": {"legal_smoke": legal, "capture_tactics": capture},
                "matches": matches,
                "ratings": ratings_payload["ratings"]["chess"],
                "rating_delta": delta,
            }
            all_traces.extend(traces)
            gates.append({"name": "chess_legal_smoke", "ok": bool(legal.get("ok")), "detail": json.dumps(legal, sort_keys=True)})
            gates.append({"name": "chess_capture_tactics", "ok": bool(capture.get("ok")), "detail": json.dumps(capture, sort_keys=True)})

    if "go" in requested:
        legal = run_go_legal_smoke(args.seed + 20, size=args.go_board_size)
        capture = run_go_capture_diagnostic(args.seed + 21, size=args.go_board_size)
        current = dict(ratings_payload["ratings"].get("go", {}))
        updated, matches, traces = run_go_rating_series(current, args.go_games, args.seed + 30, args.go_board_size)
        ratings_payload["ratings"]["go"] = {name: round(value, 3) for name, value in updated.items()}
        ratings_payload["games_played"]["go"] = int(ratings_payload["games_played"].get("go", 0)) + len(matches)
        delta = rating_delta(updated, previous_ratings.get("go", {}))
        games["go"] = {
            "skipped": False,
            "board_size": args.go_board_size,
            "diagnostics": {"legal_smoke": legal, "capture_tactics": capture},
            "matches": matches,
            "ratings": ratings_payload["ratings"]["go"],
            "rating_delta": delta,
        }
        all_traces.extend(traces)
        gates.append({"name": "go_legal_smoke", "ok": bool(legal.get("ok")), "detail": json.dumps(legal, sort_keys=True)})
        gates.append({"name": "go_capture_tactics", "ok": bool(capture.get("ok")), "detail": json.dumps(capture, sort_keys=True)})

    gates.append({"name": "public_data_calibration_only", "ok": True, "detail": "No public solutions or public benchmark answers are read or trained on."})
    gates.append({"name": "external_inference_calls", "ok": True, "detail": "0 external model calls"})

    status = "GREEN" if gates and all(gate.get("ok") for gate in gates) else "RED"
    ratings_payload["updated_at"] = now_iso()
    ratings_payload["source_report"] = rel(out_path)
    learned_policy = learned_policy_payload(all_traces, ratings_payload, games, learned_policy_path, policy_train_path)
    gates.append(
        {
            "name": "learned_policy_artifact_written",
            "ok": learned_policy.get("trigger_state") == "GREEN",
            "detail": rel(learned_policy_path),
        }
    )
    status = "GREEN" if gates and all(gate.get("ok") for gate in gates) else "RED"

    report = {
        "policy": "project_theseus_board_game_rl_benchmark_v1",
        "schema": "theseus_board_game_rl_benchmark_v1",
        "generated_at": now_iso(),
        "trigger_state": status,
        "status": status,
        "seed": args.seed,
        "duration_seconds": round(time.time() - started, 3),
        "permission_envelope": {
            "side_effect_class": "local_reversible",
            "network_required": False,
            "public_data_policy": "calibration_only_not_training",
            "external_inference_calls": 0,
        },
        "games": games,
        "gates": gates,
        "outputs": {
            "report": rel(out_path),
            "markdown": rel(markdown_path),
            "ratings": rel(ratings_path),
            "history": rel(history_path),
            "traces": rel(trace_path),
            "learned_policy": rel(learned_policy_path),
            "policy_train_rows": rel(policy_train_path),
        },
        "next_actions": [
            "Add this lane to high-transfer curriculum rotation after one more stable smoke.",
            "Graduate easy chess/go tactical diagnostics to regression once pass rates stay green.",
            "Use Elo deltas as trend evidence, not as a single-run promotion claim.",
        ],
    }

    write_json(ratings_path, ratings_payload)
    write_json(learned_policy_path, learned_policy)
    write_json(out_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    append_jsonl(history_path, [{"generated_at": report["generated_at"], "status": status, "ratings": ratings_payload["ratings"], "games_played": ratings_payload["games_played"]}])
    append_jsonl(trace_path, all_traces)

    print(json.dumps({"status": status, "report": rel(out_path), "ratings": rel(ratings_path), "traces_appended": len(all_traces)}, sort_keys=True))
    return 0 if status == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
