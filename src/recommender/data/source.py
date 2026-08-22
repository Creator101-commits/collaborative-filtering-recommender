"""Typed parsers for the MovieLens 1M source files."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class SourceDataError(ValueError):
    """Raised when a MovieLens source record is malformed."""


@dataclass(frozen=True, slots=True)
class Rating:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True, slots=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    gender: str
    age_group: int
    occupation: int
    zip_code: str


def _fields(line: str, count: int, record_type: str) -> list[str]:
    fields = line.rstrip("\r\n").split("::")
    if len(fields) != count or any(not field for field in fields):
        raise SourceDataError(f"invalid {record_type} record")
    return fields


def _positive_int(value: str, field: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise SourceDataError(f"{field} must be an integer") from error
    if number <= 0:
        raise SourceDataError(f"{field} must be positive")
    return number


def parse_rating(line: str) -> Rating:
    user_id, movie_id, rating_text, timestamp = _fields(line, 4, "rating")
    rating = _positive_int(rating_text, "rating")
    if rating > 5:
        raise SourceDataError("rating must be between 1 and 5")
    return Rating(
        user_id=_positive_int(user_id, "user ID"),
        movie_id=_positive_int(movie_id, "movie ID"),
        rating=rating,
        timestamp=_positive_int(timestamp, "timestamp"),
    )


def parse_movie(line: str) -> Movie:
    movie_id, title, genres_text = _fields(line, 3, "movie")
    genres = tuple(genres_text.split("|"))
    if any(not genre for genre in genres):
        raise SourceDataError("movie genres must be non-empty")
    return Movie(movie_id=_positive_int(movie_id, "movie ID"), title=title, genres=genres)


def parse_user(line: str) -> User:
    user_id, gender, age_text, occupation_text, zip_code = _fields(line, 5, "user")
    if gender not in {"F", "M"}:
        raise SourceDataError("gender must be F or M")
    age_group = _positive_int(age_text, "age group")
    if age_group not in {1, 18, 25, 35, 45, 50, 56}:
        raise SourceDataError("unknown age group")
    try:
        occupation = int(occupation_text)
    except ValueError as error:
        raise SourceDataError("occupation must be an integer") from error
    if occupation not in range(21):
        raise SourceDataError("occupation must be between 0 and 20")
    return User(
        user_id=_positive_int(user_id, "user ID"),
        gender=gender,
        age_group=age_group,
        occupation=occupation,
        zip_code=zip_code,
    )


def count_records(path: Path, parser: Callable[[str], object]) -> int:
    count = 0
    with path.open(encoding="latin-1", newline="") as source:
        for count, line in enumerate(source, start=1):
            try:
                parser(line)
            except SourceDataError as error:
                raise SourceDataError(f"{path.name}:{count}: {error}") from error
    if count == 0:
        raise SourceDataError(f"{path.name} is empty")
    return count
