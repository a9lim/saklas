"""The geometry instrument: a thin adapter over the unified ``Monitor``.

The geometry family's engine — whitened subspace reads, the four
conditional capture modes' scoring entry points, flat-batched and curved
foot-solve paths — stays in ``core/monitor.py``/``core/monitor_attach.py``
untouched: the capture modes are session/HiddenCapture state and the
Monitor is an established engine, so folding it into the instrument
abstraction would combine two independent risks (the orchestration
extraction and an engine rewrite) for no architectural reward.  This
adapter gives the family the same face as the lens/SAE instruments —
attach/detach/specs, gate-channel capabilities, probe-hash identity — so
the session can expose one ``instruments`` registry and the HTTP layer
can enumerate families uniformly.
"""

from __future__ import annotations

import hashlib
from typing import Any, TYPE_CHECKING

import torch

from saklas.core.instruments.types import (
    Assignment,
    Axis,
    Distance,
    Fraction,
    GateRef,
    Membership,
    validate_gate_channels,
)

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession


class GeometryInstrument:
    """Session-lifetime handle for the whitened-geometry read family."""

    family = "geometry"

    #: Every gate channel: axes, fraction, membership, label distance,
    #: soft assignment — the full whitened-reading key family.
    _GATE_CHANNELS: tuple[type, ...] = (
        Axis, Fraction, Membership, Distance, Assignment,
    )

    def __init__(self, session: "SaklasSession") -> None:
        self._session = session

    # -------------------------------------------------------------- registry

    def attach(
        self,
        selector: str,
        *,
        as_name: str | None = None,
        top_n: int = 3,
    ) -> str:
        """Attach a manifold probe (any shape — a 2-node concept axis is
        the rank-1 case) to the unified Monitor.

        Probe attach loads the manifold onto the model device and builds
        device-resident whitened factors — GPU work that must not run
        concurrently with another model op; held under the exclusive
        ``_model_exclusive`` section like the other families' attaches.
        Cache invalidation stays at the session's ``add_probe`` boundary.
        """
        session = self._session
        name = as_name if as_name is not None else selector
        with session._model_exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            # Manifold reads are Mahalanobis-only. Build the neutral artifact
            # under this same exclusive section so a concurrent generation
            # cannot race it.
            _ = session.whitener
            manifold = session._resolve_probe_manifold(selector)
            session._monitor.add_probe(name, manifold, top_n=top_n)
        return name

    def detach(self, name: str) -> None:
        self._session._monitor.remove_probe(name)

    @property
    def names(self) -> list[str]:
        return list(self._session._monitor.probe_names)

    def specs(self) -> dict[str, dict[str, Any]]:
        """Attached probe spec snapshots — the geometry analogue of the
        lens/SAE spec dicts (manifold identity + shape flags; the full
        wire info shape stays in ``server/probe_routes``)."""
        out: dict[str, dict[str, Any]] = {}
        for name, probe in self._session._monitor.attached_probes().items():
            out[name] = {
                "manifold": probe.manifold.name,
                "top_n": int(probe.top_n),
                "is_affine": bool(probe.is_affine),
                "layers": sorted(probe.manifold.layers),
            }
        return out

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    def probe_hash(self, name: str) -> str | None:
        """sha256 of the probe's baked tensor bytes (deterministic across
        machines/devices; fp32-normalized).  A 2-node concept hashes its
        folded baked-direction view for continuity with the pre-coords
        scalar monitor's drift check; a multi-node / curved probe hashes
        the per-layer subspace geometry directly."""
        session = self._session
        manifold = session._monitor.manifolds.get(name)
        if manifold is None:
            return None
        from saklas.core.capture import folded_directions

        h = hashlib.sha256()
        try:
            profile = folded_directions(manifold)
            per_layer: dict[int, list[torch.Tensor]] = {
                L: [profile[L]] for L in profile
            }
        except ValueError:
            per_layer = {}
            for layer_idx, sub in manifold.layers.items():
                tensors = [sub.mean, sub.basis]
                if sub.node_coords is not None:
                    tensors.append(sub.node_coords)
                per_layer[layer_idx] = tensors
        for layer_idx in sorted(per_layer.keys()):
            for tensor in per_layer[layer_idx]:
                arr = tensor.detach().to("cpu").to(torch.float32).contiguous()
                h.update(arr.numpy().tobytes())
        return h.hexdigest()


__all__ = ["GeometryInstrument"]
