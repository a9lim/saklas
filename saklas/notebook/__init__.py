"""Plotly-based notebook helpers for saklas results.

Bundles four figure functions and a DataFrame coercion helper that turn the
existing structured types (``Profile``, ``ResultCollector``,
``ProbeReadings``) into interactive plotly figures suitable for Jupyter,
HTML reports, and PNG export via kaleido.

Imports of plotly / pandas are gated behind each public function — calling
into the module without ``saklas[notebook]`` installed raises
:class:`NotebookExtraNotInstalled` with the install hint.

Public surface:

* :func:`plot_alpha_sweep` — alpha → probe means / secondary metric (dual y-axis)
* :func:`plot_probe_correlation` — N×N magnitude-weighted cosine heatmap
* :func:`plot_layer_norms` — per-layer ``||baked||`` bar chart
* :func:`plot_trait_history` — per-probe ``per_generation`` timeline
* :func:`to_dataframe` — coerce a ``ResultCollector`` or list of results to ``pd.DataFrame``
"""
from __future__ import annotations

from saklas.notebook.plots import (
    NotebookExtraNotInstalled,
    plot_alpha_sweep,
    plot_layer_norms,
    plot_probe_correlation,
    plot_trait_history,
)
from saklas.notebook.data import to_dataframe

__all__ = [
    "NotebookExtraNotInstalled",
    "plot_alpha_sweep",
    "plot_layer_norms",
    "plot_probe_correlation",
    "plot_trait_history",
    "to_dataframe",
]
