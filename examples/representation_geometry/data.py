"""Probe definitions for the representation-geometry examples.

Each probe is a templated discover manifold: a slot, a set of values, and a few
neutral elicitation contexts. ``author.py`` turns one into a template + a
``fit_mode=auto`` discover manifold; ``analyze.py`` reads the fitted layout.

Four probes, chosen to contrast:
  - ``countries``         — entity names. Negative control: the centroid reads
                            the *name token*, which is lexically organised, so
                            geography barely shows.
  - ``years``             — an ordinal vocabulary under neutral "name a year"
                            framing. Order is recoverable but smeared across many
                            dimensions.
  - ``years_now``         — the same years framed as the *current* year. The
                            deixis-to-now collapses the code to a clean ~1-D line.
  - ``years_now_future``  — current-year framing extended past the present, so the
                            recency line runs into a cliff at the model's "now".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    slot: str
    kind: str                       # "geographic" | "ordinal"
    values: tuple[str, ...]
    contexts: tuple[dict[str, str], ...]
    description: str
    framing: str = ""               # human label for plot titles
    geo: dict[str, tuple[float, float, str]] | None = None  # countries only
    future: bool = False            # ordinal: run the future-cliff analysis

    def canonical_contexts(self) -> list[dict[str, object]]:
        """The ``{turns, assistant}`` shape ``create_template_folder`` wants."""
        return [
            {"turns": [{"role": "user", "content": c["user"]}],
             "assistant": c["assistant"]}
            for c in self.contexts
        ]


# --------------------------------------------------------------------------- #
# countries — approximate centroids (lat N+, lon E+) + a coarse region tag. The
# region is used only to colour scatters; it is never fed to the fit.
# --------------------------------------------------------------------------- #
COUNTRIES: dict[str, tuple[float, float, str]] = {
    "France": (46.6, 2.2, "W.Europe"), "Germany": (51.2, 10.5, "W.Europe"),
    "Italy": (41.9, 12.6, "S.Europe"), "Spain": (40.5, -3.7, "S.Europe"),
    "Portugal": (39.4, -8.2, "S.Europe"), "United Kingdom": (54.0, -2.0, "W.Europe"),
    "Ireland": (53.4, -8.2, "W.Europe"), "Netherlands": (52.1, 5.3, "W.Europe"),
    "Belgium": (50.5, 4.5, "W.Europe"), "Switzerland": (46.8, 8.2, "W.Europe"),
    "Austria": (47.5, 14.6, "C.Europe"), "Poland": (51.9, 19.1, "C.Europe"),
    "Sweden": (60.1, 18.6, "N.Europe"), "Norway": (60.5, 8.5, "N.Europe"),
    "Finland": (61.9, 25.7, "N.Europe"), "Greece": (39.1, 21.8, "S.Europe"),
    "Russia": (61.5, 105.3, "Eurasia"), "Ukraine": (48.4, 31.2, "E.Europe"),
    "Turkey": (39.0, 35.2, "MidEast"), "Israel": (31.0, 34.9, "MidEast"),
    "Saudi Arabia": (23.9, 45.1, "MidEast"), "Iran": (32.4, 53.7, "MidEast"),
    "Iraq": (33.2, 43.7, "MidEast"), "Egypt": (26.8, 30.8, "N.Africa"),
    "Morocco": (31.8, -7.1, "N.Africa"), "Nigeria": (9.1, 8.7, "Sub-Sahara"),
    "Kenya": (-0.0, 37.9, "Sub-Sahara"), "Ethiopia": (9.1, 40.5, "Sub-Sahara"),
    "South Africa": (-30.6, 22.9, "Sub-Sahara"), "Ghana": (7.9, -1.0, "Sub-Sahara"),
    "India": (20.6, 79.0, "S.Asia"), "Pakistan": (30.4, 69.3, "S.Asia"),
    "Bangladesh": (23.7, 90.4, "S.Asia"), "Afghanistan": (33.9, 67.7, "S.Asia"),
    "China": (35.9, 104.2, "E.Asia"), "Japan": (36.2, 138.3, "E.Asia"),
    "South Korea": (35.9, 127.8, "E.Asia"), "North Korea": (40.3, 127.5, "E.Asia"),
    "Mongolia": (46.9, 103.8, "E.Asia"), "Thailand": (15.9, 101.0, "SE.Asia"),
    "Vietnam": (14.1, 108.3, "SE.Asia"), "Indonesia": (-0.8, 113.9, "SE.Asia"),
    "Philippines": (12.9, 121.8, "SE.Asia"), "Malaysia": (4.2, 102.0, "SE.Asia"),
    "Singapore": (1.4, 103.8, "SE.Asia"), "Australia": (-25.3, 133.8, "Oceania"),
    "New Zealand": (-40.9, 174.9, "Oceania"), "United States": (37.1, -95.7, "N.America"),
    "Canada": (56.1, -106.3, "N.America"), "Mexico": (23.6, -102.6, "N.America"),
    "Brazil": (-14.2, -51.9, "S.America"), "Argentina": (-38.4, -63.6, "S.America"),
    "Chile": (-35.7, -71.5, "S.America"), "Peru": (-9.2, -75.0, "S.America"),
    "Colombia": (4.6, -74.3, "S.America"), "Cuba": (21.5, -77.8, "Caribbean"),
}

_NAME_A_COUNTRY = [
    {"user": "Name a country.", "assistant": "[C]"},
    {"user": "Name any country in the world.", "assistant": "[C]"},
    {"user": "What's a country you can think of?", "assistant": "[C]"},
    {"user": "Tell me the name of a country.", "assistant": "[C]"},
    {"user": "Pick a country, anywhere.", "assistant": "[C]"},
]

_NAME_A_YEAR = [
    {"user": "Name a year.", "assistant": "[Y]"},
    {"user": "Pick a year, any year.", "assistant": "[Y]"},
    {"user": "Tell me a year.", "assistant": "[Y]"},
    {"user": "What year comes to mind?", "assistant": "[Y]"},
    {"user": "Give me a year from history.", "assistant": "[Y]"},
]

# Current-year deixis. Note the label rule (a node label must start with a letter,
# so a bare "1985" is illegal): the assertion is folded into the *value*
# ("the current year is 1985"), which keeps the label letter-initial and is
# byte-identical across nodes (cancels as common-mode), while the digit tokens are
# still computed under the assertion frame.
_CURRENT_YEAR = [
    {"user": "What year is it?", "assistant": "[Y]"},
    {"user": "What is the current year?", "assistant": "[Y]"},
    {"user": "What year are we in?", "assistant": "[Y]"},
    {"user": "What's the year right now?", "assistant": "[Y]"},
    {"user": "If someone asked you the year, what would you say?", "assistant": "[Y]"},
]


def _year_values(lo: int, hi: int, fmt: str) -> tuple[str, ...]:
    return tuple(fmt.format(y) for y in range(lo, hi + 1))


PROBES: dict[str, Probe] = {
    "countries": Probe(
        name="countries", slot="[C]", kind="geographic",
        values=tuple(COUNTRIES), contexts=tuple(_NAME_A_COUNTRY), geo=COUNTRIES,
        framing="name-a-country",
        description="Country names — does conceptual proximity recover geography?",
    ),
    "years": Probe(
        name="years", slot="[Y]", kind="ordinal",
        values=_year_values(1900, 2020, "the year {}"), contexts=tuple(_NAME_A_YEAR),
        framing="name-a-year",
        description="Years under neutral framing — what shape does time make?",
    ),
    "years_now": Probe(
        name="years_now", slot="[Y]", kind="ordinal",
        values=_year_values(1900, 2020, "the current year is {}"),
        contexts=tuple(_CURRENT_YEAR), framing="current-year deixis",
        description="Years framed as the current year — deixis-to-now A/B vs `years`.",
    ),
    "years_now_future": Probe(
        name="years_now_future", slot="[Y]", kind="ordinal",
        values=_year_values(1900, 2035, "the current year is {}"),
        contexts=tuple(_CURRENT_YEAR), framing="current-year deixis", future=True,
        description="Current-year framing past the present — locating the model's 'now'.",
    ),
}
