"""Which (track, ycset, polarity) samples exist and how many files each has.

This is the table that used to be built by ``init_data()`` in make_snapshot.py.
Keeping it here means stage 1 has no embedded data, and the same inventory can
be queried by any other tool.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

TRACKS: List[str] = ["ll", "dd"]
POLARITIES: List[str] = ["magup", "magdown"]
YCSETS: List[str] = ["25c1", "25c2", "25c3", "25c4", "24c2a", "24c3a", "24c4a"]

# Raw inventory: track -> polarity -> "<ycset>_<nfiles>-<ycset>_<nfiles>-..."
_RAW: Dict[str, Dict[str, str]] = {
    "ll": {
        "magup": "25c4_80-25c3_156-25c1_95-24c4a_33-24c3a_83-24c2a_240",
        "magdown": "25c4_59-25c2_246-25c1_22-24c4a_52-24c3a_69-24c2a_72",
    },
    "dd": {
        "magup": "25c4_188-25c3_320-25c1_426-24c4a_132-24c3a_305-24c2a_57",
        "magdown": "25c4_126-25c2_1007-25c1_100-24c4a_240-24c3a_250-24c2a_14",
    },
}


def _parse() -> Dict[Tuple[str, str, str], int]:
    out: Dict[Tuple[str, str, str], int] = {}
    for track, by_pol in _RAW.items():
        for polarity, spec in by_pol.items():
            for item in spec.split("-"):
                ycset, n = item.rsplit("_", 1)
                out[(track, ycset, polarity)] = int(n)
    return out


#: (track, ycset, polarity) -> number of available input files
INVENTORY: Dict[Tuple[str, str, str], int] = _parse()


def n_files(track: str, ycset: str, polarity: str) -> Optional[int]:
    return INVENTORY.get((track, ycset, polarity))


def iter_samples(
    track: Optional[str] = None,
    ycset: Optional[str] = None,
    polarity: Optional[str] = None,
) -> Iterator[Tuple[str, str, str, int]]:
    """Yield (track, ycset, polarity, n_files) for every sample matching the
    filters.  Filters left as None match everything, which is how stage 1
    turns a partially-specified CLI request into a concrete work list."""
    for (t, y, p), n in sorted(INVENTORY.items()):
        if track is not None and t != track:
            continue
        if ycset is not None and y != ycset:
            continue
        if polarity is not None and p != polarity:
            continue
        yield t, y, p, n