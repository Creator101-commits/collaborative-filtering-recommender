"""Tests for MovieLens source parsing."""

from collections.abc import Callable
from pathlib import Path

import pytest

from recommender.data.source import (
    SourceDataError,
    count_records,
    parse_movie,
    parse_rating,
    parse_user,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ml-1m"


@pytest.mark.parametrize(
    ("filename", "parser"),
    [
        ("movies.dat", parse_movie),
        ("ratings.dat", parse_rating),
        ("users.dat", parse_user),
    ],
)
def test_fixture_records_are_valid(filename: str, parser: Callable[[str], object]) -> None:
    assert count_records(FIXTURE_DIR / filename, parser) == 2


@pytest.mark.parametrize(
    ("parser", "record"),
    [
        (parse_movie, "1::Missing genres::"),
        (parse_rating, "1::2::6::978300760"),
        (parse_user, "1::X::25::1::48067"),
    ],
)
def test_malformed_records_are_rejected(parser: Callable[[str], object], record: str) -> None:
    with pytest.raises(SourceDataError):
        parser(record)
