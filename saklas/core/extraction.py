"""ManifoldExtractionPipeline — fit a steering manifold from an authored corpus.

The single extraction pipeline (4.0): a steering vector is the 2-node ``pca``
case of a manifold, so concept extraction (``session.extract``) and manifold
fitting share this one pipeline.  It pools per-node centroids, fits the
per-layer PCA (+ RBF for curved manifolds), bakes the per-layer share, and
writes the per-model tensor.

Dependencies are passed structurally (not as a back-reference to the session)
via the runtime-checkable :class:`ModelHandle` protocol plus an
:class:`EventBus` for ``ManifoldExtracted`` emission.  ``SaklasSession``
satisfies the protocol implicitly, so construction reads as
``ManifoldExtractionPipeline(self, self.events)``.

The session gates re-entry against ``GenState.IDLE`` before forwarding; the
pipeline itself does not touch generation state.
"""
from __future__ import annotations

import pathlib
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from saklas.core.events import EventBus, ManifoldExtracted
from saklas.core.sae import SaeBackend
from saklas.io.paths import tensor_filename


# ----------------------------------------------------------------------
# Structural protocols.  Runtime-checkable so callers (and tests) can
# ``isinstance(session, ModelHandle)`` to sanity-check the implicit
# implementation.  ``SaklasSession`` satisfies each by virtue of carrying
# the listed attributes / methods.
# ----------------------------------------------------------------------

@runtime_checkable
class ModelHandle(Protocol):
    """Read-only surface the pipeline needs from the live HF session.

    Held as a *handle*, not a copy — the pipeline must see the same
    model object the session uses; otherwise device-side state diverges.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def tokenizer(self) -> Any: ...  # PreTrainedTokenizerBase

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def layers(self) -> Any: ...  # ``get_layers`` returns ``nn.ModuleList`` — list-like

    def _run_generator(
        self, system_msg: str, prompt: str, max_new_tokens: int,
    ) -> str:
        """Single-turn LLM call backing conversational corpus generation.

        Underscore-prefixed because the override site is per-session
        (subclass-and-override is the established test pattern).  The
        protocol shape mirrors the existing ``SaklasSession._run_generator``
        signature exactly so the session satisfies it implicitly.
        """
        ...

    def generate_responses(
        self,
        concepts: list[str],
        kinds: list[str | None],
        *,
        roles: dict[str, str | None] | None = None,
        samples_per_prompt: int = ...,
        max_new_tokens: int = ...,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]: ...


def _diagnostics_to_dict(diag: Any) -> dict[str, Any]:
    """Convert a discover-mode diagnostics dataclass into a JSON-safe dict.

    Both :class:`PcaDiagnostics` and :class:`SpectralDiagnostics` carry
    one or more tensor fields; the sidecar is JSON, so tensors are
    converted to plain Python lists and floats here.  The dispatcher is
    structural — duck-typed on the dataclass fields — so adding a third
    method later doesn't require touching this helper as long as the
    dataclass stays JSON-serializable in this way.
    """
    out: dict[str, Any] = {}
    for name in (
        "per_component_variance", "cumulative_variance", "eigenvalues",
    ):
        if hasattr(diag, name):
            t = getattr(diag, name)
            if isinstance(t, torch.Tensor):
                out[name] = [float(x) for x in t.tolist()]
            else:
                out[name] = list(t)
    for name in (
        "picked_k", "gap_index", "k_nn", "component_count",
    ):
        if hasattr(diag, name):
            out[name] = int(getattr(diag, name))
    for name in ("threshold", "gap_magnitude", "bandwidth"):
        if hasattr(diag, name):
            out[name] = float(getattr(diag, name))
    return out


class ManifoldExtractionPipeline:
    """Fit an RBF-based steering manifold from an authored corpus.

    Distinct from :class:`ExtractionPipeline` — manifold extraction has a
    fundamentally different input (N labeled node groups, no contrastive
    pairs, no scenario generation), so it is its own pipeline rather than
    a method on the concept extractor.  It reuses the :class:`ModelHandle`
    protocol (it needs ``model`` / ``tokenizer`` / ``layers`` / ``device``
    / ``model_id``) and emits :class:`ManifoldExtracted`.

    Feature space: ``sae=None`` fits per-layer PCA + RBF directly on
    residual-stream centroids.  ``sae="<release>"`` reconstructs each
    centroid through the SAE (encode then decode) before the fit — a
    denoised, sparse-feature-supported centroid — and restricts the
    manifold to the SAE's covered layers.  Either way the fitted
    :class:`~saklas.core.manifold.LayerSubspace` is model-space, so the
    steering hook never touches the SAE.
    """

    __slots__ = ("_events", "_handle")

    def __init__(self, model_handle: ModelHandle, events: EventBus) -> None:
        self._handle = model_handle
        self._events = events

    def fit(
        self,
        folder: str | pathlib.Path,
        *,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ):
        """Fit (or load from cache) a manifold for the session's model.

        ``folder`` is a manifold pack directory — either authored (the
        user supplied a domain spec + per-node coordinates) or
        discover-mode (the user supplied only labeled node corpora; the
        coords are derived per-model via
        :func:`saklas.core.manifold.discover_coords`).  Returns a
        :class:`~saklas.core.manifold.Manifold`.

        A cache hit — the per-model tensor exists and its sidecar
        ``nodes_sha256`` still matches the folder's current state —
        short-circuits the forward passes.  For discover-mode folders
        the staleness key folds in ``fit_mode`` + ``hyperparams``, so
        a refit with different hyperparameters reliably misses cache.
        """
        from saklas.core.manifold import (
            CustomDomain,
            Manifold,
            compute_node_centroid,
            discover_coords,
            domain_from_spec,
            fit_affine_subspace,
            fit_layer_subspace,
            invert_parameterization,
            load_manifold,
            save_manifold,
            subspace_share,
        )
        from saklas.core.vectors import compute_dls_axes
        from saklas.io.manifolds import (
            ManifoldFolder, ManifoldSidecar, min_nodes,
        )
        from saklas.core.errors import SaeCoverageError

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        mf = ManifoldFolder.load(pathlib.Path(folder))
        node_groups = mf.node_groups()
        nodes_sha = mf.nodes_sha256()

        # Resolve the SAE backend once (lazy import — non-SAE callers
        # never touch the SAE layer).
        sae_backend: SaeBackend | None
        sae_release: str | None
        if sae is None:
            sae_backend = None
            sae_release = None
        elif isinstance(sae, str):
            from saklas.core.sae import load_sae_backend
            sae_backend = load_sae_backend(
                sae,
                revision=sae_revision,
                model_id=self._handle.model_id,
                device=self._handle.device,
            )
            sae_release = sae
        else:
            sae_backend = sae
            sae_release = sae.release

        tensor_path = pathlib.Path(folder) / tensor_filename(
            self._handle.model_id, release=sae_release,
        )

        # Cache hit: tensor present + every fit-affecting input unchanged.
        # ``nodes_sha256`` folds in the corpus, plus either the domain
        # spec + node coords (authored) or the fit_mode + hyperparams
        # (discover); ``sae_revision`` is the SAE the centroids are
        # reconstructed through and does not ride the filename, so it is
        # checked here or a stale tensor is served.
        sidecar_path = tensor_path.with_suffix(".json")
        cached_revision = (
            sae_backend.revision if sae_backend is not None else None
        )
        if tensor_path.exists() and sidecar_path.exists():
            try:
                sc = ManifoldSidecar.load(sidecar_path)
            except (KeyError, ValueError):
                sc = None
            if (
                sc is not None
                and sc.nodes_sha256 == nodes_sha
                and sc.sae_revision == cached_revision
            ):
                _progress(f"Loaded cached manifold '{mf.name}'.")
                manifold = load_manifold(tensor_path)
                self._events.emit(ManifoldExtracted(
                    name=mf.name, manifold=manifold,
                    metadata=dict(manifold.metadata),
                ))
                return manifold

        model = self._handle.model
        tokenizer = self._handle.tokenizer
        layers = self._handle.layers
        device = self._handle.device
        n_layers = len(layers)
        K = len(node_groups)

        # ``model_type`` is the family-key for the role-header registry —
        # only needed when any node carries a custom ``role``.  We resolve
        # it once up front and pass it on each centroid call regardless;
        # ``_encode_and_capture_all`` only consults it when ``role`` is set.
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        node_roles = mf._roles_padded()
        node_kinds = mf._kinds_padded()
        any_role = any(r is not None for r in node_roles)
        any_kind = any(k is not None for k in node_kinds)
        if any_role and model_type is None:
            raise ValueError(
                f"manifold {mf.name!r} carries per-node roles but model "
                f"config has no 'model_type' — cannot resolve the role-header "
                f"registry entry"
            )

        # SAE coverage — fail-fast (mirrors the vector path in
        # ``vectors._capture_diffs_for_pairs``, which raises
        # ``SaeCoverageError`` *before* its forward loop).  Validate the
        # fit-layer set here, before the expensive per-node centroid
        # pooling, so an SAE release that covers none of the model's
        # layers errors immediately instead of after K node passes.
        if sae_backend is not None:
            fit_layers = sorted(
                set(sae_backend.layers) & set(range(n_layers))
            )
            if not fit_layers:
                raise SaeCoverageError(
                    f"SAE release {sae_backend.release!r} covers no layers "
                    f"of {self._handle.model_id}"
                )
            feature_space = f"sae-{sae_backend.release}"
        else:
            fit_layers = list(range(n_layers))
            feature_space = "raw"

        # 1. Per-node centroids (one forward pass per response) — shared
        #    between authored and discover paths.  Conversational (4.0 / A2):
        #    each node corpus is a list of in-character responses to the shared
        #    baseline prompts, pooled as ``[user: prompt, assistant: response]``
        #    pairs (response[i] -> prompt[i % k]).  Per-node role rides through
        #    ``compute_node_centroid`` only when set (persona-baselined fit);
        #    a ``None`` role pools under the standard assistant (swap-back)
        #    baseline.
        from saklas.core.vectors import _load_baseline_prompts
        baseline_prompts = _load_baseline_prompts()
        per_node: list[dict[int, torch.Tensor]] = []
        for (label, responses), role in zip(node_groups, node_roles):
            role_note = f" [role={role}]" if role else ""
            _progress(
                f"Pooling node '{label}'{role_note} "
                f"({len(responses)} responses)..."
            )
            per_node.append(compute_node_centroid(
                model, tokenizer, layers, device, responses, baseline_prompts,
                role=role, model_type=model_type,
            ))

        # 1b. Monopolar (concept-vs-neutral).  A 1-node ``pca`` folder has no
        #     second pole to span an affine subspace, so it can only mean one
        #     thing: the concept against the **neutral baseline**.  Its steering
        #     direction is the displacement of the single concept centroid from
        #     the model's neutral activation mean ν (``layer_means``) — sourced
        #     fresh per model, never a stored corpus — folded into a 1-node
        #     neutral-anchored ray via :func:`fold_directions_to_subspace` (the
        #     same primitive ``merge`` uses).  No discover-coords, no DLS, no
        #     synthetic second node; ``concept − ν`` already cancels common-mode
        #     like DiM, so the raw δ̂ basis is appropriate.  The engine
        #     recognizes the shape structurally (a flat ``pca`` fit otherwise
        #     needs ``k + 1 ≥ 2`` poised nodes).
        if mf.fit_mode == "pca" and K == 1:
            from saklas.core.vectors import fold_directions_to_subspace
            means = getattr(self._handle, "layer_means", None) or {}
            if not means:
                raise ValueError(
                    f"monopolar manifold {mf.name!r} (a 1-node concept-vs-"
                    f"neutral fit) needs the model's neutral activation mean "
                    f"(layer_means) as its negative pole, but none is "
                    f"available; regenerate the neutral corpus so layer_means "
                    f"can build"
                )
            _wh = getattr(self._handle, "whitener", None)
            maha = (
                _wh if _wh is not None and _wh.covers_all(fit_layers) else None
            )
            concept_label = node_groups[0][0]
            directions: dict[int, torch.Tensor] = {}
            for idx in fit_layers:
                nu = means.get(idx)
                if nu is None:
                    continue  # no neutral anchor at this layer → can't fold it
                c = per_node[0][idx].to(torch.float32).reshape(-1)
                if sae_backend is not None:
                    with torch.no_grad():
                        feat = sae_backend.encode_layer(
                            idx, c.reshape(1, -1).to(device),
                        )
                        c = sae_backend.decode_layer(idx, feat).detach().to(
                            "cpu", torch.float32,
                        ).reshape(-1)
                directions[idx] = c - nu.to(torch.float32).reshape(-1)
            _progress(
                f"Folding monopolar '{concept_label}' − neutral across "
                f"{len(directions)} layers..."
            )
            manifold = fold_directions_to_subspace(
                mf.name, directions, means, whitener=maha,
                label=concept_label, feature_space=feature_space,
            )
            # Carry the per-node role/kind onto the fitted ray so steer-time
            # role substitution (``nearest_node_role``) and provenance work; the
            # fold primitive itself is role-agnostic.
            manifold.node_roles = list(node_roles)
            manifold.node_kinds = list(node_kinds)
            metadata: dict[str, Any] = {
                "method": (
                    "manifold_monopolar_sae" if sae_backend is not None
                    else "manifold_monopolar"
                ),
                "nodes_sha256": nodes_sha,
                "monopolar": True,
                # The fold keeps the raw δ̂ basis (``concept − ν`` cancels
                # common-mode like DiM), so the subspace is Euclidean; only the
                # share is whitened when the whitener covers the layers.
                "share_metric": manifold.metadata.get(
                    "share_metric", "euclidean",
                ),
                "subspace_metric": "euclidean",
            }
            if sae_backend is not None:
                metadata["sae_release"] = sae_backend.release
                metadata["sae_revision"] = sae_backend.revision
            if any_role:
                metadata["node_roles"] = list(node_roles)
            if any_kind:
                metadata["node_kinds"] = list(node_kinds)
            save_manifold(manifold, tensor_path, metadata)
            manifold.metadata.update(metadata)
            mf.write_metadata()
            self._events.emit(ManifoldExtracted(
                name=mf.name, manifold=manifold, metadata=metadata,
            ))
            return manifold

        # 2. Layer set is already resolved above (fail-fast SAE coverage).
        # No DLS analogue here on purpose: manifolds have no pos/neg
        # polarity, so Dang & Ngo's opposite-sign discriminative test is
        # undefined.  Per-layer signal is instead captured continuously by
        # the apply-time ``share_L = ||eval_rbf(node_params)||_F`` weighting
        # (low-spread layers are down-weighted, not hard-dropped).

        # 3. Resolve domain + node_params — the only step that differs
        #    between authored and discover paths.
        discover_metadata: dict[str, Any] = {}
        # ``max_subspace_dim`` caps the per-layer PCA subspace in
        # ``fit_layer_subspace`` (default :data:`DEFAULT_N_COMPONENTS` = 64).
        # Smaller values constrain the dim count that ``subspace_inject``
        # displaces at steer time — finer-grained steering control at the
        # cost of representing less per-layer activation variance.  At K
        # large (e.g. 100), the default 64 makes the per-layer subspace
        # cover most of the centroid span; reducing to the manifold's
        # intrinsic dim (=picked_k for discover) gives an order-of-
        # magnitude smaller per-α displacement, widening the coherence
        # regime.  Authored manifolds don't currently route hyperparams
        # so they inherit the default.
        max_subspace_dim_override: int | None = None

        if mf.fit_mode == "authored":
            domain = domain_from_spec(mf.domain)
            node_coords = torch.tensor(mf.node_coords, dtype=torch.float32)
            node_params = domain.embed(node_coords)
            method = "manifold_sae" if sae_backend is not None else "manifold_pca"
        else:
            # Discover: derive coords from per-node centroids at a
            # reference layer (middle of the fit-layer set by default,
            # or ``reference_layer`` override).  The same coords feed
            # the per-layer RBF as the manifold's intrinsic coordinates
            # — wrapped in a ``CustomDomain(k)`` with identity embed so
            # the existing fit machinery handles them unchanged.
            hyperparams = dict(mf.hyperparams)
            ref_layer = hyperparams.pop(
                "reference_layer", fit_layers[len(fit_layers) // 2],
            )
            # ``max_subspace_dim`` is consumed by the per-layer fit, not
            # by ``discover_coords`` — pop it before the discover call so
            # the dispatcher doesn't get an unexpected kwarg.
            if "max_subspace_dim" in hyperparams:
                max_subspace_dim_override = int(hyperparams.pop("max_subspace_dim"))
            # ``anchor_origin`` is the post-discover origin-anchoring
            # toggle (see ``_HYPERPARAMS_BY_MODE`` in ``io/manifolds.py``
            # for the semantics).  We pop it here so the discover
            # dispatcher doesn't see it, and apply the translation after
            # ``discover_coords`` returns.  Spectral mode rejects it at
            # the sanitization gate; assert here for defense in depth.
            anchor_origin_raw = hyperparams.pop("anchor_origin", None)
            if anchor_origin_raw is not None and mf.fit_mode != "pca":
                raise ValueError(
                    f"discover manifold {mf.name!r}: anchor_origin is "
                    f"only supported for fit_mode='pca' (got {mf.fit_mode!r})"
                )
            anchor_label: str | None
            if anchor_origin_raw in (None, False):
                anchor_label = None
            elif anchor_origin_raw is True:
                anchor_label = "default"
            elif isinstance(anchor_origin_raw, str) and anchor_origin_raw:
                anchor_label = anchor_origin_raw
            else:
                raise ValueError(
                    f"discover manifold {mf.name!r}: anchor_origin must "
                    f"be a bool or non-empty string label, got "
                    f"{anchor_origin_raw!r}"
                )
            if ref_layer not in fit_layers:
                raise ValueError(
                    f"discover manifold {mf.name!r}: reference_layer "
                    f"{ref_layer} not in fit_layers {fit_layers}"
                )
            ref_stack = torch.stack(
                [per_node[k][ref_layer] for k in range(K)]
            )  # (K, D) fp32 CPU
            if sae_backend is not None:
                with torch.no_grad():
                    feat = sae_backend.encode_layer(
                        ref_layer, ref_stack.to(device),
                    )
                    recon = sae_backend.decode_layer(ref_layer, feat)
                ref_stack = recon.detach().to("cpu", torch.float32)
            _progress(
                f"Deriving {mf.fit_mode} coords from layer "
                f"{ref_layer} ({K} centroids)..."
            )
            derived_coords, diagnostics = discover_coords(
                ref_stack, method=mf.fit_mode, **hyperparams,
            )
            k = derived_coords.shape[1]
            # Node-count floor.  The curved (spectral) path fits an RBF
            # surface and needs the poisedness floor ``min_nodes(k) = 2k+1``.
            # The flat (``pca``) path fits an *affine* subspace with no RBF —
            # it only needs ``k+1`` affinely-independent centroids to span a
            # k-dim subspace, so a rank-1 (k=1) fit is valid at K=2: that is a
            # difference-of-means steering vector (ARCHITECTURE §1/§5, "a
            # vector = a 2-node fit_mode=pca folder").
            if mf.fit_mode == "pca":
                floor = k + 1
                floor_reason = "to span the affine subspace"
            else:
                floor = min_nodes(k)
                floor_reason = "for the RBF fit"
            if floor > K:
                raise ValueError(
                    f"discover manifold {mf.name!r}: picked k={k} needs "
                    f">= {floor} nodes {floor_reason}, got K={K}"
                )
            # Origin anchoring (pca-discover only).  Find the anchor
            # node by label, take its row from ``derived_coords`` as the
            # pre-translate authoring coord, and shift every coord by
            # ``-anchor_coord`` — the anchor lands at ``(0, ..., 0)`` in
            # the user-facing coord system.  RBF interpolation is exact
            # at fit nodes, so steering ``<manifold>%0,...,0`` reproduces
            # the anchor's per-layer behavior (its corpus pools to the
            # value the interpolant emits at origin).
            anchor_authoring_coord: list[float] | None = None
            if anchor_label is not None:
                node_labels_for_anchor = [
                    label for label, _ in node_groups
                ]
                if anchor_label not in node_labels_for_anchor:
                    raise ValueError(
                        f"discover manifold {mf.name!r}: anchor_origin "
                        f"requires a node labeled {anchor_label!r}, but "
                        f"the corpus has nodes "
                        f"{node_labels_for_anchor!r}"
                    )
                anchor_idx = node_labels_for_anchor.index(anchor_label)
                anchor_coord = derived_coords[anchor_idx].clone()
                anchor_authoring_coord = anchor_coord.tolist()
                derived_coords = derived_coords - anchor_coord
            domain = CustomDomain(k)
            node_coords = derived_coords
            node_params = derived_coords  # identity embedding
            method = (
                "manifold_discover_sae" if sae_backend is not None
                else f"manifold_discover_{mf.fit_mode}"
            )
            # Record what we ran with — fit_mode + hyperparams (with
            # any derived defaults filled in, e.g. spectral's resolved
            # ``k_nn``/``bandwidth``) + the diagnostics for the sidecar
            # and the inspector surfaces.
            resolved_hyperparams = dict(hyperparams)
            resolved_hyperparams["reference_layer"] = int(ref_layer)
            if max_subspace_dim_override is not None:
                resolved_hyperparams["max_subspace_dim"] = max_subspace_dim_override
            if hasattr(diagnostics, "k_nn"):  # SpectralDiagnostics
                resolved_hyperparams["k_nn"] = int(diagnostics.k_nn)  # pyright: ignore[reportAttributeAccessIssue]  # SpectralDiagnostics only; guarded by hasattr
                resolved_hyperparams["bandwidth"] = float(
                    diagnostics.bandwidth,  # pyright: ignore[reportAttributeAccessIssue]  # SpectralDiagnostics only; guarded by hasattr
                )
            if anchor_label is not None:
                resolved_hyperparams["anchor_origin"] = anchor_label
            discover_metadata = {
                "fit_mode": mf.fit_mode,
                "hyperparams": resolved_hyperparams,
                "diagnostics": _diagnostics_to_dict(diagnostics),
            }
            if anchor_label is not None and anchor_authoring_coord is not None:
                # Stamp the pre-translate authoring coord into metadata
                # so post-hoc inspection can verify the translation and
                # measure how far the anchor sat from the K-centroid mean
                # in the original PCA space.  Not consumed by any runtime
                # path — purely diagnostic.
                discover_metadata["anchor_pre_translate_coord"] = (
                    anchor_authoring_coord
                )

        # 4. Per-layer fit.  Stack centroids -> (K, D); for the SAE
        #    variant reconstruct through the SAE before fitting.
        #
        # Mahalanobis share weighting (parity with the vector bake metric):
        # the whitener is resolved here, *after* the cache-hit return and
        # the fail-fast SAE/role checks, since ``handle.whitener`` can
        # trigger a lazy neutral-activation build — the same deferral the
        # vector pipeline applies to its bake whitener.  We only bake the
        # whitened share when the whitener covers *every* fit layer, so the
        # apply-time cross-layer normalization compares like with like;
        # partial coverage (or a handle without ``.whitener`` — CPU test
        # stubs) leaves the dict empty and the apply path falls back to the
        # Euclidean centroid-spread.  The basis is model-space for both raw
        # and SAE fits (centroids are decoded back before the fit), so the
        # residual-stream whitener applies to the SAE variant unchanged.
        _whitener = getattr(self._handle, "whitener", None)
        maha_whitener = (
            _whitener
            if _whitener is not None and _whitener.covers_all(fit_layers)
            else None
        )
        _progress(
            f"Fitting RBF interpolant across {len(fit_layers)} layers..."
        )
        layer_subs = {}
        explained_variance: dict[int, float] = {}
        mahalanobis_share: dict[int, float] = {}
        # Neutral baseline (probe-centering means), **ungated** by the
        # whitener: the curved fit's neutral-anchor (``mean = P_basis(ν)``,
        # §5) consumes it.  ``None`` on a CPU-stub handle ⇒ the fit falls back
        # to the centroid-mean anchor.  (Also drives the origin foot below.)
        _handle_means = getattr(self._handle, "layer_means", None)
        fit_kwargs: dict[str, Any] = {}
        if max_subspace_dim_override is not None:
            fit_kwargs["n_components"] = max_subspace_dim_override

        def _stacked_centroids(idx: int) -> torch.Tensor:
            s = torch.stack([per_node[k][idx] for k in range(K)])  # (K, D) fp32 CPU
            if sae_backend is not None:
                with torch.no_grad():
                    feat = sae_backend.encode_layer(idx, s.to(device))
                    recon = sae_backend.decode_layer(idx, feat)
                s = recon.detach().to("cpu", torch.float32)
            return s

        def _neutral_for(idx: int) -> "torch.Tensor | None":
            return (
                _handle_means[idx]
                if _handle_means is not None and idx in _handle_means
                else None
            )

        def _bake_share(
            idx: int, sub: Any, mu_coords: torch.Tensor,
        ) -> None:
            # Per-layer budget = the **μ-centered** (anchor-independent)
            # whitened spread (§5 / ``subspace_share``): the share measures
            # signal spread, not where neutral sits.  Gated on the covers-all
            # whitener; absent ⇒ Euclidean spread fallback at apply.
            if maha_whitener is None:
                return
            mahalanobis_share[idx] = subspace_share(
                mu_coords, sub.basis, whitener=maha_whitener, layer=idx,
            )

        if mf.fit_mode == "pca":
            # FLAT affine fit — the discover-``pca`` path is a flat rank-``k``
            # subspace, not an RBF surface (ARCHITECTURE §1/§5).  Per layer:
            # ``fit_affine_subspace`` (μ-centered PCA basis at the derived
            # intrinsic dim ``k``, neutral-anchored frame, real per-layer node
            # coords).  **Whitened/Fisher basis** when the whitener covers every
            # fit layer (Step 8 made this unconditional — the de-rogued basis is
            # what makes the no-lever gain coherent; same ``covers_all`` gate as
            # the curved path + the Mahalanobis share), Euclidean on the
            # no-coverage fallback.  The shared ``node_coords`` stays the derived
            # PCA layout (display/labels); the real per-layer steer coords live
            # on each ``LayerSubspace.node_coords``.
            affine_kwargs = dict(fit_kwargs)
            # rank = the derived intrinsic dim (the shared layout's width — set
            # for every fit_mode, so always bound here unlike the discover-only
            # ``k``); ``max_subspace_dim`` override in ``fit_kwargs`` wins.
            affine_kwargs.setdefault("n_components", int(node_coords.shape[1]))
            raw_fits: dict[int, tuple[Any, torch.Tensor]] = {}
            stacks: dict[int, torch.Tensor] = {}
            for idx in fit_layers:
                stacked = _stacked_centroids(idx)
                stacks[idx] = stacked
                sub, mu_coords, ev_ratio = fit_affine_subspace(
                    stacked, neutral_mean=_neutral_for(idx),
                    whitener=maha_whitener, layer=idx,
                    orient_to=0, **affine_kwargs,
                )
                raw_fits[idx] = (sub, mu_coords)
                explained_variance[idx] = ev_ratio
            # Per-axis DLS straddle over all fit layers at once (flat → DLS;
            # the global all-fail fallback matches the folded-vector path).
            dls_kept = compute_dls_axes(
                {idx: stacks[idx] for idx in raw_fits},
                {idx: raw_fits[idx][0].basis for idx in raw_fits},
                _handle_means,
            )
            for idx, (sub, mu_coords) in raw_fits.items():
                kept = sorted(dls_kept.get(idx, set()))
                if not kept:
                    continue  # no axis straddles the baseline → drop the layer
                if len(kept) < sub.rank:
                    sub = sub.select_axes(kept)
                    mu_coords = mu_coords[:, kept]
                layer_subs[idx] = sub
                _bake_share(idx, sub, mu_coords)
        else:
            # CURVED (authored / spectral): RBF surface fit, neutral-anchored.
            for idx in fit_layers:
                stacked = _stacked_centroids(idx)
                # ``neutral_mean`` neutral-anchors the frame; ``maha_whitener``
                # still selects the (whitened/Fisher) basis on the curved path
                # — see the Step-3 basis-metric note (curved keeps its existing
                # whitened selection pending the Step 8 gate decision).
                sub, ev_ratio = fit_layer_subspace(
                    stacked, node_params,
                    whitener=maha_whitener, layer=idx,
                    neutral_mean=_neutral_for(idx), **fit_kwargs,
                )
                layer_subs[idx] = sub
                explained_variance[idx] = ev_ratio
                # μ-centered share (NOT ``eval_rbf(node_params)`` — the surface
                # is neutral-anchored, so its node values aren't μ-centered).
                mu_centered = stacked.to(torch.float32)
                mu_centered = mu_centered - mu_centered.mean(dim=0)
                mu_coords = mu_centered @ sub.basis.to(torch.float32).T  # (K, R)
                _bake_share(idx, sub, mu_coords)

        # Origin ``O_L`` — the per-layer foot of the neutral mean on ``M``, in
        # authoring coords ``(n,)``.  **Curved only** — a flat affine subspace's
        # surface fills its span, so neutral's foot is reduced-coord 0 (the
        # ``!`` ablation target) with no stored origin (§2); routing an affine
        # subspace through ``invert_parameterization`` would also hit
        # ``rbf_params()`` and raise.  Each layer's cold-start foot seed; layers
        # whose mean isn't resolvable (CPU stub) are simply absent.
        origin: dict[int, torch.Tensor] = {}
        if mf.fit_mode != "pca" and _handle_means is not None:
            for idx, sub in layer_subs.items():
                if idx not in _handle_means:
                    continue
                mu = _handle_means[idx].to(
                    device="cpu", dtype=torch.float32,
                ).reshape(-1)
                # Reduced coords of the neutral mean in this layer's subspace,
                # then nearest-point invert onto ``M``.
                q = (mu - sub.mean) @ sub.basis.T  # (R,)
                O_coords, _dist = invert_parameterization(
                    sub, domain, q, node_coords,
                )
                origin[idx] = O_coords.reshape(-1).to(torch.float32)

        manifold = Manifold(
            name=mf.name,
            domain=domain,
            node_labels=[label for label, _ in node_groups],
            node_coords=node_coords,
            layers=layer_subs,
            feature_space=feature_space,
            node_roles=list(node_roles),
            node_kinds=list(node_kinds),
            explained_variance=explained_variance,
            mahalanobis_share=mahalanobis_share,
            origin=origin,
        )

        # 5. Persist + refresh the folder integrity manifest.
        metadata: dict[str, Any] = {
            "method": method,
            "nodes_sha256": nodes_sha,
            # Which metric the per-layer share weighting used — the
            # manifold analogue of the vector sidecar's ``bake`` field.
            # "mahalanobis" iff the whitened share was baked (whitener
            # covered every fit layer); "euclidean" on fallback.
            "share_metric": (
                "mahalanobis" if maha_whitener is not None else "euclidean"
            ),
            # Which metric the per-layer PCA *subspace selection* used —
            # "mahalanobis" => whitened/Fisher (de-rogued directions) when the
            # whitener covers every fit layer, else "euclidean".  Both the flat
            # (``pca``) and curved paths whiten under the same ``covers_all``
            # gate as the share (Step 8 made the flat path unconditional), so
            # subspace_metric and share_metric agree.
            "subspace_metric": (
                "mahalanobis" if maha_whitener is not None else "euclidean"
            ),
        }
        if sae_backend is not None:
            metadata["sae_release"] = sae_backend.release
            metadata["sae_revision"] = sae_backend.revision
        if any_role:
            # Per-node roles ride into the sidecar so `manifold
            # show` and the inspector surfaces can report "this node was
            # pooled as <role>" without re-reading manifold.json.  The
            # order matches ``node_labels``; a missing entry means
            # ``None`` (standard assistant baseline).
            metadata["node_roles"] = list(node_roles)
        if any_kind:
            # Per-node kind rides into the sidecar for inspector/provenance,
            # ``node_labels`` order; absent when no node carries a kind.
            metadata["node_kinds"] = list(node_kinds)
        metadata.update(discover_metadata)
        save_manifold(manifold, tensor_path, metadata)
        manifold.metadata.update(metadata)
        mf.write_metadata()

        self._events.emit(ManifoldExtracted(
            name=mf.name, manifold=manifold, metadata=metadata,
        ))
        return manifold
