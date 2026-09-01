"""``RunID`` — the identity of a single analysis stream.

Every stage is keyed on the same four things: track, ycset, polarity and HLT1
selection.  The old code encoded these into filenames like
``dd_25c4_magup_188.root`` and then re-parsed them by splitting on '_', which
broke as soon as a field was optional or a name contained an underscore
(``Pip_Hlt1TrackMVADecision_TOS`` certainly does).

Here the identity is an object, the file *stem* carries only the three fields
that are always present, and everything else (HLT1, file counts, tags) lives in
directory structure and metadata where it cannot be mangled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Directory name used when no HLT1 requirement is applied.
NO_HLT1 = "nohlt1"


@dataclass(frozen=True, order=True)
class RunID:
    track: str
    ycset: str
    polarity: str
    hlt1: str = ""

    def __post_init__(self) -> None:
        if self.track not in ("ll", "dd"):
            raise ValueError(f"Unexpected track {self.track!r}: use 'll' or 'dd'.")
        if self.polarity not in ("magup", "magdown"):
            raise ValueError(
                f"Unexpected polarity {self.polarity!r}: use 'magup' or 'magdown'."
            )

    # ---- naming ------------------------------------------------------
    @property
    def key(self) -> str:
        """Stem used for skim files and per-run directories."""
        return f"{self.track}_{self.ycset}_{self.polarity}"

    @property
    def hlt1_dir(self) -> str:
        """Directory-safe name of the HLT1 selection."""
        return self.hlt1 if self.hlt1 else NO_HLT1

    @property
    def slug(self) -> str:
        """Fully-qualified, human-readable identifier including HLT1."""
        return f"{self.key}__{self.hlt1_dir}"

    def label(self) -> str:
        return f"{self.track} {self.ycset} {self.polarity} [{self.hlt1_dir}]"

    # ---- serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "track": self.track,
            "ycset": self.ycset,
            "polarity": self.polarity,
            "hlt1": self.hlt1,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunID":
        return cls(
            track=d["track"],
            ycset=d["ycset"],
            polarity=d["polarity"],
            hlt1=d.get("hlt1", "") or "",
        )

    @classmethod
    def from_key(cls, key: str, hlt1: str = "") -> "RunID":
        """Parse ``<track>_<ycset>_<polarity>``.  Strict: exactly three fields."""
        parts = key.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Cannot parse RunID from {key!r}; expected "
                "'<track>_<ycset>_<polarity>'"
            )
        track, ycset, polarity = parts
        return cls(track=track, ycset=ycset, polarity=polarity, hlt1=hlt1)


def skipped(runid: RunID) -> bool:
    """Sample combinations that are known not to exist in the data.

    Previously this was an unexplained ``if`` in the middle of the stage-1
    loop; keeping it as a named predicate makes the exclusion visible and
    testable.
    """
    return (runid.polarity == "magdown" and runid.ycset == "25c3") or (
        runid.polarity == "magup" and runid.ycset == "25c2"
    )