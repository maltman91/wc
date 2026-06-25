from typing import NamedTuple


class GroupStandingRow(NamedTuple):
    Group: str
    Pos: int
    Team: str
    Pts: int
    GD: int
    GF: int
    GA: int
    W: int
    D: int
    L: int
    P: int


__all__ = ["GroupStandingRow"]
