"""Leakage-controlled full-catalog candidate generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def candidate_items(item_count: int, seen_items: Iterable[int]) -> tuple[int, ...]:
    """Return every catalog item except the supplied observed items."""
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    seen = set(seen_items)
    if any(item < 0 or item >= item_count for item in seen):
        raise ValueError("seen item is outside the catalog")
    return tuple(item for item in range(item_count) if item not in seen)


def _row_index(row: Mapping[str, Any], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"interaction has no valid {field}") from error
    return value


def generate_candidate_catalog(
    item_count: int,
    observed_interactions: Iterable[Mapping[str, Any]],
    *,
    user_indices: Iterable[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Build candidates using only interactions explicitly passed as observed.

    The caller controls the information set: for validation this must be the
    training partition, so validation and test interactions remain candidates.
    """
    seen_by_user: defaultdict[int, set[int]] = defaultdict(set)
    for row in observed_interactions:
        user_idx = _row_index(row, "user_idx")
        item_idx = _row_index(row, "item_idx")
        if user_idx < 0:
            raise ValueError("user_idx must be non-negative")
        if not 0 <= item_idx < item_count:
            raise ValueError("item_idx is outside the catalog")
        seen_by_user[user_idx].add(item_idx)

    if user_indices is None:
        users = set(seen_by_user)
    else:
        users = set(user_indices)
        if any(user_idx < 0 for user_idx in users):
            raise ValueError("user_idx must be non-negative")
    return {
        user_idx: candidate_items(item_count, seen_by_user.get(user_idx, set()))
        for user_idx in sorted(users)
    }


build_candidate_catalog = generate_candidate_catalog
