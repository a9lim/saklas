"""ExtractionPipeline — concept-extraction orchestration lifted out of session.py.

The pipeline owns the full custom-concept flow: tensor cache → curated or
local ``statements.json`` (reused by default) → generate scenarios → save
→ generate pairs → save → contrastive PCA → save tensor.  Curated concepts
save under ``default/<c>/``; user concepts under ``local/<c>/``.

Dependencies are passed structurally (not as a back-reference to the
session) via two runtime-checkable Protocols — :class:`ModelHandle` and
:class:`PackWriter` — plus an :class:`EventBus` for ``VectorExtracted``
emission.  ``SaklasSession`` implements both protocols implicitly, so
construction reads as ``ExtractionPipeline(self, self, self.events)``.

The session gates re-entry against ``GenState.IDLE`` before forwarding;
the pipeline itself does not touch generation state.  See ``Phase 7`` of
``docs/plans/audit-followups.md`` for the design rationale.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import torch

from saklas.core.events import EventBus, VectorExtracted
from saklas.core.profile import Profile
from saklas.core.sae import SaeBackend
from saklas.io.datasource import DataSource
from saklas.io.packs import hash_file
from saklas.io.paths import tensor_filename
from saklas.core.vectors import (
    # Imported for ``_extractor_for``'s ``globals()`` lookup, which is
    # the deliberate dispatch pattern that lets test monkeypatches at
    # module scope reach the dispatcher.  Direct name references would
    # bypass the indirection.
    extract_contrastive,  # noqa: F401
    extract_difference_of_means,  # noqa: F401
    save_profile as _save_profile,
    load_profile as _load_profile,
    load_contrastive_pairs,
)


# Default extraction method for fresh extractions.  v2.1 flips the
# default from contrastive PCA to difference-of-means (Im & Li 2025);
# ``--method pca`` and the matching API kwarg keep the legacy path
# accessible for direct A/B comparisons and reproducing v1.x results.
DEFAULT_EXTRACTION_METHOD: Literal["dim", "pca"] = "dim"


def _method_label(
    method: Literal["dim", "pca"], sae_backend: SaeBackend | None,
) -> str:
    """Sidecar ``method`` label for an extraction.

    DiM extractions write ``"difference_of_means"`` (raw) or
    ``"dim_sae"`` (SAE feature space).  PCA extractions preserve the
    pre-v2.1 labels ``"contrastive_pca"`` / ``"pca_center_sae"`` so
    older readers and on-disk tensors round-trip unchanged.
    """
    if method == "dim":
        return "dim_sae" if sae_backend is not None else "difference_of_means"
    if method == "pca":
        return "pca_center_sae" if sae_backend is not None else "contrastive_pca"
    raise ValueError(
        f"unknown extraction method {method!r} (expected 'dim' | 'pca')"
    )


def _extractor_for(
    method: Literal["dim", "pca"],
):
    """Return the per-method low-level extractor function.

    Resolves through the module namespace (``globals()``) rather than the
    closed-over import binding so test monkeypatches at module scope reach
    the dispatcher.
    """
    if method == "dim":
        return globals()["extract_difference_of_means"]
    if method == "pca":
        return globals()["extract_contrastive"]
    raise ValueError(
        f"unknown extraction method {method!r} (expected 'dim' | 'pca')"
    )


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
        """Single-turn LLM call shared by scenario and pair generators.

        Underscore-prefixed because the override site is per-session
        (subclass-and-override is the established test pattern).  The
        protocol shape mirrors the existing ``SaklasSession._run_generator``
        signature exactly so the session satisfies it implicitly.
        """
        ...

    def generate_scenarios(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = ...,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]: ...

    def generate_pairs(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = ...,
        *,
        scenarios: list[str] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[tuple[str, str]]: ...


@runtime_checkable
class PackWriter(Protocol):
    """Local-pack mutations the pipeline performs."""

    def _local_concept_folder(self, canonical: str) -> pathlib.Path:
        """Return ``local/<canonical>/`` with a placeholder ``pack.json``."""
        ...

    def _promote_profile(
        self, profile: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Cast a freshly-loaded profile onto the session's device + dtype."""
        ...

    def _update_local_pack_files(self, folder: pathlib.Path) -> None:
        """Recompute ``pack.json.files`` after writing tensors / statements."""
        ...


# ----------------------------------------------------------------------
# Pipeline.
# ----------------------------------------------------------------------


class ExtractionPipeline:
    """Self-contained concept-extraction pipeline.

    Construction accepts the structural dependencies extraction actually
    uses; the pipeline holds none of them as inherited "session" state.
    See module docstring for the audit reference.
    """

    __slots__ = ("_handle", "_packs", "_events")

    def __init__(
        self,
        model_handle: ModelHandle,
        pack_writer: PackWriter,
        events: EventBus,
    ) -> None:
        self._handle = model_handle
        self._packs = pack_writer
        self._events = events

    # -- public entry point ------------------------------------------------

    def extract(
        self,
        source,
        baseline: str | None = None,
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
        method: Literal["dim", "pca"] = DEFAULT_EXTRACTION_METHOD,
        dls: bool = True,
    ) -> tuple[str, Profile]:
        """Extract a steering vector profile and emit ``VectorExtracted``.

        Mirrors the historical ``SaklasSession.extract`` signature: ``sae``
        accepts a release-name string (resolved via :func:`load_sae_backend`)
        or an already-resolved :class:`SaeBackend` for direct injection.

        ``method`` selects the per-layer direction algorithm:

        - ``"dim"`` (default, v2.1+) — difference-of-means; provably
          optimal for the linear-steering objective (Im & Li 2025).
        - ``"pca"`` — legacy contrastive PCA; first principal component
          of the diffs.  Retained for backward compatibility and
          side-by-side comparison via the ``:pca`` selector variant.

        Cache-hit semantics:

        - Tensor cache hits short-circuit and emit ``VectorExtracted``
          with ``method`` set to the prior extraction method.
        - On tensor miss, ``statements.json`` is reused when present
          unless ``force_statements=True`` or explicit ``scenarios=[...]``
          were supplied.
        - On statements miss, the full pipeline runs end-to-end.
        """
        canonical, profile = self._extract_impl(
            source, baseline,
            scenarios=scenarios,
            reuse_scenarios=reuse_scenarios,
            force_statements=force_statements,
            on_progress=on_progress,
            sae=sae,
            sae_revision=sae_revision,
            namespace=namespace,
            method=method,
            dls=dls,
        )
        try:
            meta = dict(profile.metadata) if hasattr(profile, "metadata") else {}
        except Exception:
            meta = {}
        self._events.emit(
            VectorExtracted(name=canonical, profile=profile, metadata=meta),
        )
        return canonical, profile

    # -- internal pipeline -------------------------------------------------

    def _extract_impl(
        self,
        source,
        baseline: str | None = None,
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
        method: Literal["dim", "pca"] = DEFAULT_EXTRACTION_METHOD,
        dls: bool = True,
    ) -> tuple[str, Profile]:
        """Extraction body.  See :meth:`extract` for the wrapper.

        Pipeline: tensor cache check → ``statements.json`` cache (curated
        or local) unless ``force_statements`` → otherwise generate
        scenarios + pairs → save both → extract contrastive → save tensor.
        Scenarios persist to ``local/<canonical>/scenarios.json`` alongside
        ``statements.json``.
        """
        # Local import to avoid an import cycle: ``session`` imports the
        # pipeline at module load, and ``canonical_concept_name`` lives on
        # the session module for back-compat with downstream callers
        # (cloning, tui, runners, selectors).  The function is pure; the
        # late import has no runtime cost after first use.
        from saklas.core.session import canonical_concept_name, _split_composite_source

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Normalize source.
        if isinstance(source, str):
            concept, baseline = _split_composite_source(source, baseline)
        elif isinstance(source, DataSource):
            concept = source.name
            baseline = None
        elif isinstance(source, list):
            concept = "custom"
            baseline = None
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        canonical = canonical_concept_name(concept, baseline)

        # Resolve the SAE backend once.  No-op when ``sae is None`` —
        # the ``load_sae_backend`` import is lazy so non-SAE callers
        # don't touch the SAE layer.  Pre-resolved ``SaeBackend`` instances
        # pass through verbatim.
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
            # Pre-resolved backend.
            sae_backend = sae
            sae_release = sae.release

        sae_metadata: dict = {}
        if sae_backend is not None:
            sae_metadata = {
                "sae_release": sae_backend.release,
                "sae_revision": sae_backend.revision,
                "sae_ids_by_layer": getattr(sae_backend, "sae_ids_by_layer", {}),
            }

        method_label = _method_label(method, sae_backend)
        extractor = _extractor_for(method)

        # Mahalanobis bake (v2.1+): DiM extraction uses the per-model
        # whitener for share allocation when available.  We pull off the
        # handle via getattr — keeps the ModelHandle protocol minimal
        # and lets test stubs that don't implement ``.whitener`` fall
        # back to Euclidean.  PCA branch ignores the whitener (it scores
        # via EVR, not magnitude).  v2.1+: layer_means + dls are
        # threaded uniformly into both extractors; the centered DLS
        # check fires when both are present.  Tests / mock stubs that
        # don't carry layer_means just keep all layers.
        bake_label = "euclidean"
        # Eager kwargs — cheap, always known.  The expensive fields
        # (``layer_means``, ``whitener``) are deferred to
        # :func:`_resolve_extract_kwargs` so a cache-hit short-circuit
        # below doesn't trigger ``handle.layer_means`` / ``handle.whitener``
        # — both can launch a lazy ``bootstrap_layer_means`` /
        # ``LayerWhitener`` build that runs forward passes through the
        # model, which is exactly the work the cache hit was supposed
        # to avoid.  Pre-v2.1 this dict was eager and
        # ``probes=[]`` sessions paid the neutral build on every
        # cache-hit ``session.extract`` call.
        eager_kwargs: dict = {
            "sae": sae_backend,
            "concept_label": canonical,
            "dls": dls,
        }

        def _resolve_extract_kwargs() -> dict:
            """Materialize the full extractor kwargs dict on demand.

            Resolves ``layer_means`` and ``whitener`` from the handle
            — both of which may trigger lazy bootstrap — and mutates
            the closed-over ``bake_label`` when the whitener loads
            successfully.  Called at every extractor call site
            *after* the cache-hit short-circuit so cache hits skip
            the bootstrap.
            """
            nonlocal bake_label
            out = dict(eager_kwargs)
            out["layer_means"] = getattr(self._handle, "layer_means", None)
            if method == "dim":
                handle_whitener = getattr(self._handle, "whitener", None)
                if handle_whitener is not None:
                    out["whitener"] = handle_whitener
                    bake_label = "mahalanobis"
            return out

        def _build_return(
            profile_dict: dict,
            diagnostics: dict[int, dict[str, float]] | None = None,
        ) -> tuple[str, Profile]:
            meta: dict = {"method": method_label, "bake": bake_label}
            meta.update(sae_metadata)
            if diagnostics:
                meta["diagnostics"] = diagnostics
            out_name = (
                canonical
                if sae_backend is None
                else f"{canonical}:sae-{sae_backend.release}"
            )
            return out_name, Profile(profile_dict, metadata=meta)

        def _save_meta(
            extra: dict | None = None,
            *,
            diagnostics: dict[int, dict[str, float]] | None = None,
        ) -> dict:
            meta: dict = {"method": method_label, "bake": bake_label}
            if extra:
                meta.update(extra)
            meta.update(sae_metadata)
            if diagnostics:
                meta["diagnostics"] = diagnostics
            return meta

        model = self._handle.model
        tokenizer = self._handle.tokenizer
        layers = self._handle.layers
        model_id = self._handle.model_id

        def _path_for(folder: pathlib.Path) -> str:
            return str(folder / tensor_filename(
                model_id, release=sae_release, method=method,
            ))

        # For DataSource or raw pairs, skip the full pipeline — just extract.
        if isinstance(source, (DataSource, list)):
            if isinstance(source, list):
                ds = DataSource(pairs=source)
            else:
                ds = source
            folder = self._packs._local_concept_folder(canonical)
            cache_path = _path_for(folder)
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._packs._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                out_name = (
                    canonical
                    if sae_backend is None
                    else f"{canonical}:sae-{sae_backend.release}"
                )
                return out_name, Profile(profile, metadata=_meta)
            except (FileNotFoundError, KeyError, ValueError):
                pass

            _progress(f"Extracting profile ({len(ds.pairs)} pairs)...")
            pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
            profile, diagnostics = extractor(
                model, tokenizer, pairs, layers=layers,
                **_resolve_extract_kwargs(),
            )
            _save_profile(profile, cache_path, _save_meta(diagnostics=diagnostics))
            self._packs._update_local_pack_files(folder)
            return _build_return(profile, diagnostics)

        # String source — full pipeline.  Pack lookup scans installed
        # namespaces, but bare names must not silently pick the first
        # duplicate.  If the caller supplies ``namespace=``, honor only that
        # namespace.
        from saklas.io.selectors import _all_concepts, AmbiguousSelectorError
        matches = [
            c for c in _all_concepts()
            if c.name == canonical and (namespace is None or c.namespace == namespace)
        ]
        if namespace is not None and not matches and namespace != "local":
            raise FileNotFoundError(
                f"concept pack '{namespace}/{canonical}' is not installed"
            )
        if namespace is None and len(matches) > 1:
            qualified = ", ".join(f"{c.namespace}/{c.name}" for c in matches)
            raise AmbiguousSelectorError(
                f"ambiguous concept '{canonical}': matches {qualified}. "
                f"Specify with a namespace."
            )
        pack_folder = matches[0].folder if matches else None

        if pack_folder is not None:
            cache_path = _path_for(pack_folder)
        else:
            cache_path = _path_for(
                pathlib.Path(self._packs._local_concept_folder(canonical))
            )

        # 1. Vector cache — short-circuits unless a regen path is requested.
        #    ``force_statements=True`` or explicit ``scenarios=[...]`` both
        #    mean the caller wants fresh pairs, which definitionally
        #    invalidates any tensor trained on the old pairs — bypassing
        #    the tensor cache here is the only semantically coherent
        #    behavior for those flags.  No cache hit means the full
        #    pipeline runs end-to-end and overwrites the stale tensor.
        if not force_statements and scenarios is None:
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._packs._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                out_name = (
                    canonical
                    if sae_backend is None
                    else f"{canonical}:sae-{sae_backend.release}"
                )
                return out_name, Profile(profile, metadata=_meta)
            except (FileNotFoundError, KeyError, ValueError):
                pass

        # 2. Installed-statements fast path — default reuses bundled / HF /
        #    local statements.json when present.  ``force_statements=True``
        #    skips this branch and falls through to regeneration.  Passing an
        #    explicit ``scenarios=`` also skips — if the caller supplied
        #    scenarios they're clearly opting into fresh pair generation.
        if pack_folder is not None and not force_statements and scenarios is None:
            pack_stmts = pack_folder / "statements.json"
            if pack_stmts.exists():
                _progress(f"Using curated statements for '{canonical}'...")
                ds = load_contrastive_pairs(str(pack_stmts))
                # ``**extract_kwargs`` carries ``sae``, ``concept_label``,
                # ``whitener`` (DiM/Mahalanobis bake), ``dls``, and
                # ``layer_means`` (DLS centering).  Earlier this site
                # passed only ``sae`` + ``concept_label`` — silently
                # dropping whitener and DLS, which made the v2.1
                # Mahalanobis bake and v2.1 DLS no-ops on bundled
                # statements paths.  Same fix applied to the local-
                # statements cache path below (site 3).
                profile, diagnostics = extractor(
                    model, tokenizer, ds["pairs"],
                    layers=layers,
                    **_resolve_extract_kwargs(),
                )
                _save_profile(profile, cache_path, _save_meta(
                    {"statements_sha256": hash_file(pack_stmts)},
                    diagnostics=diagnostics,
                ))
                self._packs._update_local_pack_files(pack_folder)
                return _build_return(profile, diagnostics)

        # 3. Local statements cache — default reuses if present.
        local_folder = self._packs._local_concept_folder(canonical)
        stmt_cache_path = str(local_folder / "statements.json")
        local_folder = pathlib.Path(stmt_cache_path).parent
        pairs = None
        if not force_statements and scenarios is None:
            try:
                with open(stmt_cache_path) as f:
                    cached = json.load(f)
                if isinstance(cached, list):
                    pairs = cached
                elif isinstance(cached, dict) and "pairs" in cached:
                    pairs = cached["pairs"]
                if pairs:
                    _progress(f"Using cached pairs for '{canonical}'.")
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                pass

        # 4. Generate scenarios + pairs if needed.
        if pairs is None:
            suffix = f" vs '{baseline}'" if baseline else ""
            local_folder.mkdir(parents=True, exist_ok=True)

            # 4a. Resolve effective scenarios.
            scn_path = local_folder / "scenarios.json"
            eff_scenarios: list[str] | None = None
            if scenarios is not None:
                eff_scenarios = list(scenarios)
                _progress(
                    f"Using {len(eff_scenarios)} caller-provided scenarios."
                )
            elif reuse_scenarios and scn_path.exists():
                try:
                    with open(scn_path) as f:
                        data = json.load(f)
                    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
                        eff_scenarios = [str(s) for s in data["scenarios"]]
                    elif isinstance(data, list):
                        eff_scenarios = [str(s) for s in data]
                    if eff_scenarios:
                        _progress(
                            f"Reusing {len(eff_scenarios)} cached scenarios "
                            f"for '{canonical}'."
                        )
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    eff_scenarios = None

            if not eff_scenarios:
                _progress(f"Generating scenarios for '{concept}'{suffix}...")
                eff_scenarios = self._handle.generate_scenarios(
                    concept, baseline, on_progress=_progress,
                )

            if not eff_scenarios:
                raise ValueError(
                    f"Could not generate scenarios for '{concept}'. "
                    f"Try a more specific concept."
                )

            # Persist scenarios to disk (overwriting any existing file).
            with open(scn_path, "w") as f:
                json.dump({"scenarios": eff_scenarios}, f, indent=2)

            _progress(
                f"Generating contrastive pairs for '{concept}'{suffix} "
                f"across {len(eff_scenarios)} domains..."
            )
            raw_pairs = self._handle.generate_pairs(
                concept, baseline,
                scenarios=eff_scenarios,
                on_progress=_progress,
            )
            pairs = [{"positive": p, "negative": n} for p, n in raw_pairs]
            with open(stmt_cache_path, "w") as f:
                json.dump(pairs, f, indent=2)

        if len(pairs) < 2:
            raise ValueError(
                f"Could only generate {len(pairs)} pairs (need >= 2). "
                f"Try a more specific concept."
            )

        # 5. Extract.  See site 2 above for why ``**extract_kwargs`` is
        # required — without it the v2.1 whitener (Mahalanobis bake)
        # and DLS keep set both silently fall through.
        _progress(
            f"Extracting {method_label} profile ({len(pairs)} pairs)..."
        )
        profile, diagnostics = extractor(
            model, tokenizer, pairs, layers=layers,
            **_resolve_extract_kwargs(),
        )
        _save_profile(profile, cache_path, _save_meta(
            {"statements_sha256": hash_file(pathlib.Path(stmt_cache_path))},
            diagnostics=diagnostics,
        ))
        self._packs._update_local_pack_files(local_folder)
        return _build_return(profile, diagnostics)
