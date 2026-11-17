"""Matrix serialization, content hashing and NumPy probability computation."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .errors import ApiError


def to_arrays(
    rain: List[List[float]], missing: List[List[int]] | None, rows: int, cols: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate shape against the registered grid and return flat float64/uint8 arrays."""
    if len(rain) != rows or any(len(r) != cols for r in rain):
        raise ApiError(422, "GRID_MISMATCH", f"rain must have shape ({rows}, {cols})")
    rain_arr = np.asarray(rain, dtype=np.float64)
    if missing is None:
        miss_arr = np.zeros((rows, cols), dtype=np.uint8)
    else:
        if len(missing) != rows or any(len(r) != cols for r in missing):
            raise ApiError(422, "GRID_MISMATCH", f"missing must have shape ({rows}, {cols})")
        miss_arr = np.asarray(missing, dtype=np.uint8)
        if np.any((miss_arr != 0) & (miss_arr != 1)):
            raise ApiError(422, "INVALID_MISSING", "missing grid must contain only 0 or 1")
    valid = miss_arr == 0
    if np.any(rain_arr[valid] < 0):
        raise ApiError(422, "NEGATIVE_RAIN", "rain values must be non-negative at valid cells")
    return rain_arr.reshape(-1), miss_arr.reshape(-1)


def content_hash(rain_flat: np.ndarray, miss_flat: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(b"rain:")
    h.update(rain_flat.astype(np.float64, copy=False).tobytes())
    h.update(b"|miss:")
    h.update(miss_flat.astype(np.uint8, copy=False).tobytes())
    return h.hexdigest()


def compute_probabilities(
    rains: Sequence[np.ndarray],
    misses: Sequence[np.ndarray],
    thresholds: Sequence[float],
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Per-cell exceedance probability over valid (non-missing) members.

    p = (#valid members with rain >= threshold) / (#valid members).
    Cells with no valid member at all get denominator 0 and a null probability.
    Returns one (probability_float64, denominator_int64) pair per threshold.
    """
    stacked_rain = np.vstack([r.reshape(1, -1) for r in rains])  # (M, N)
    stacked_miss = np.vstack([m.reshape(1, -1) for m in misses]).astype(bool)
    valid_mask = ~stacked_miss  # (M, N)
    denom = valid_mask.sum(axis=0).astype(np.int64)  # (N,)
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    safe_denom = np.where(denom > 0, denom, 1).astype(np.float64)
    for threshold in thresholds:
        hits = np.where(valid_mask, stacked_rain >= float(threshold), False)
        count = hits.sum(axis=0).astype(np.float64)
        prob = count / safe_denom
        # cells with zero valid members are represented as NaN -> JSON null
        prob = np.where(denom > 0, prob, np.nan)
        out[float(threshold)] = (prob.astype(np.float64), denom.copy())
    return out


def compute_loo_sensitivity(
    rains: Sequence[np.ndarray],
    misses: Sequence[np.ndarray],
    threshold: float,
    alert_probability: float,
    orig_prob: np.ndarray,
    orig_denom: np.ndarray,
    cols: int,
) -> List[Dict[str, Any]]:
    """Leave-one-out sensitivity over the current product's member grids.

    For every (member, cell) the exceedance probability is recomputed with that
    member removed and the warning state (``prob >= alert_probability``) is
    compared against the current product's stored probability.

    Attribution rules:
      * a member that is missing at a cell never contributed there, so it is
        not attributed for that cell;
      * cells whose original probability is null (all members missing) are
        never judged;
      * cells whose leave-one-out set is entirely missing are never judged.

    ``orig_prob`` / ``orig_denom`` are the flat probability/denominator grids
    stored in the current product for ``threshold``. Returns one record per
    (cell, member) flip, ordered by (row, col, member_idx).
    """
    stacked_rain = np.vstack([r.reshape(1, -1) for r in rains])  # (M, N)
    stacked_miss = np.vstack([m.reshape(1, -1) for m in misses]).astype(bool)
    valid = ~stacked_miss  # (M, N)
    hits = valid & (stacked_rain >= float(threshold))  # (M, N)
    total_hits = hits.sum(axis=0)  # (N,)

    orig_denom = orig_denom.astype(np.int64)
    orig_valid = orig_denom > 0  # (N,)
    orig_warn = np.zeros(orig_denom.shape, dtype=bool)
    # orig_prob is NaN exactly where orig_valid is False; mask before comparing
    orig_warn[orig_valid] = orig_prob[orig_valid] >= float(alert_probability)

    loo_denom = orig_denom.reshape(1, -1) - valid.astype(np.int64)  # (M, N)
    loo_hits = total_hits.reshape(1, -1) - hits.astype(np.int64)  # (M, N)
    loo_valid = loo_denom > 0
    safe_denom = np.where(loo_valid, loo_denom, 1).astype(np.float64)
    loo_prob = loo_hits.astype(np.float64) / safe_denom  # masked where invalid
    loo_warn = np.zeros(loo_prob.shape, dtype=bool)
    loo_warn[loo_valid] = loo_prob[loo_valid] >= float(alert_probability)

    flippable = valid & orig_valid.reshape(1, -1) & loo_valid
    flip_mask = flippable & (loo_warn != orig_warn.reshape(1, -1))

    member_ids, cell_ids = np.nonzero(flip_mask)
    records: List[Dict[str, Any]] = []
    for mi, ci in sorted(zip(member_ids.tolist(), cell_ids.tolist()), key=lambda t: (t[1], t[0])):
        row, col = divmod(ci, cols)
        records.append(
            {
                "row": row,
                "col": col,
                "member_idx": mi,
                "original_probability": float(orig_prob[ci]),
                "loo_probability": float(loo_prob[mi, ci]),
                "original_warning": bool(orig_warn[ci]),
                "loo_warning": bool(loo_warn[mi, ci]),
                "original_valid_members": int(orig_denom[ci]),
                "loo_valid_members": int(loo_denom[mi, ci]),
            }
        )
    return records


def matrix_to_json(flat: np.ndarray, rows: int, cols: int) -> List[List[float | int | None]]:
    grid = flat.reshape(rows, cols)
    result: List[List[float | int | None]] = []
    for row in grid:
        out_row: List[float | int | None] = []
        for v in row:
            fv = float(v)
            if np.isnan(fv):
                out_row.append(None)
            else:
                out_row.append(fv)
        result.append(out_row)
    return result


def encode_blob(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()


def decode_float(blob: bytes, n_cells: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64, count=n_cells).copy()


def decode_int(blob: bytes, n_cells: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.int64, count=n_cells).copy()


def decode_mask(blob: bytes, n_cells: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.uint8, count=n_cells).copy()
