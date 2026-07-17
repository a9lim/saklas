"""Transcript export / import for loom sessions.

A **transcript** is a saved path through the loom tree: system prompt
+ every user turn + every assistant turn's :class:`Recipe` (steering,
sampling, seed, probe set, per-probe content hash) + final aggregate
readings.  Serializes to YAML; round-trips through
:meth:`Transcript.to_yaml` / :meth:`Transcript.from_yaml`.

The per-node thing remains :class:`Recipe`; the file/export concept is
:class:`Transcript` so docs and CLI do not overload the two artifacts.

Schema (v2 — the cast model)::

    saklas_transcript: 2
    model_id: <hf-id>
    system_prompt: <str>
    cast:                  # optional; the tree's cast roster
      <label>: {recipe: {...}, notes: <str>}
    probes:
      - name: <probe>
        sha256: <hex>
    turns:
      - role: user|assistant     # the seat (system turns are skipped on import)
        speaker: <label>         # optional; the turn's cast label
        text: <str>
        thinking: <str>          # optional; the turn's thinking block
        recipe: {...}            # generated turns (either seat); provenance
        readings: {...}          # generated turns

On import, a turn with a ``recipe`` re-attaches as a *generated* node in its
recorded seat (provenance = recipe presence, the cast model's invariant); a
recipe-less turn is a committed one. Cast entries merge into the
session tree's roster; a label the live roster already holds with a
*different* member is left alone and flagged in the guard notes.

Three import modes (decision 11):

- **default** — attaches as a new top-level branch off the tree root.
- **here** — attaches as a child of the active node.
- **merge** — walks the active path from root, finds the deepest
  matching **user-turn** prefix between the active path and the
  transcript, attaches the non-matching tail there.  Falls back to
  root-attach when no user-turn prefix matches.

Guards:

- model mismatch → warn, refuse ``--merge``
  (:class:`TranscriptModelMismatch`), allow other modes with a banner
  on the imported root.
- system-prompt mismatch → warn, proceed, banner on imported root.
- missing probes → warn, readings recorded as-imported (display-only).
- probe hash drift → warn; ``--strict`` raises
  :class:`TranscriptProbeDriftError` on any hash mismatch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from saklas.core.errors import SaklasError
from saklas.core.loom import CastMember, LoomTree, Recipe


SAKLAS_TRANSCRIPT_VERSION = 2

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranscriptError(SaklasError):
    """Base for transcript-IO errors."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class TranscriptFormatError(TranscriptError):
    """Raised when a transcript YAML can't be parsed / lacks required fields."""


class TranscriptModelMismatch(TranscriptError):
    """Raised when a transcript's model differs from the session and ``--merge`` is requested."""


class TranscriptProbeDriftError(TranscriptError):
    """Raised under ``strict=True`` when any probe sha256 differs from the session."""


ImportMode = Literal["default", "here", "merge"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeRef:
    """Probe entry in the transcript header — name + hash for drift detection."""

    name: str
    sha256: str


@dataclass
class Turn:
    """One conversation turn captured for replay.

    ``role`` is the *seat*; ``speaker`` the cast label the turn was
    rendered with (``None`` = the family's standard label).  Provenance
    is recipe presence — a generated turn carries its :class:`Recipe`
    whatever its seat.
    """

    role: Literal["user", "assistant", "system"]
    text: str
    speaker: str | None = None
    thinking: str | None = None
    recipe: Recipe | None = None
    readings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "text": self.text}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        if self.thinking is not None:
            out["thinking"] = self.thinking
        if self.recipe is not None:
            out["recipe"] = self.recipe.to_dict()
        if self.readings:
            out["readings"] = dict(self.readings)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Turn":
        recipe = None
        rec = data.get("recipe")
        if rec is not None:
            recipe = Recipe.from_dict(dict(rec))
        readings: dict[str, float] = {}
        for name, val in (data.get("readings") or {}).items():
            try:
                readings[str(name)] = float(val)
            except (TypeError, ValueError):
                continue
        speaker = data.get("speaker")
        thinking = data.get("thinking")
        return cls(
            role=str(data.get("role", "user")),  # pyright: ignore[reportArgumentType]  # str narrowed to Literal at runtime
            text=str(data.get("text", "")),
            speaker=str(speaker) if speaker is not None else None,
            thinking=str(thinking) if thinking is not None else None,
            recipe=recipe,
            readings=readings,
        )


@dataclass
class Transcript:
    """A saved path through the tree.

    Build from a tree node via :meth:`from_path`; round-trip through
    :meth:`to_yaml` / :meth:`from_yaml`; import into a live session
    via :meth:`import_into`.
    """

    model_id: str | None
    system_prompt: str | None
    probes: list[ProbeRef]
    turns: list[Turn]
    # Cast roster (v2): label → member, mirrored from the tree at export.
    cast: dict[str, CastMember] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction from a session path
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, node_id: str | None, session: Any) -> "Transcript":
        """Build a transcript from the path ending at ``node_id``.

        ``node_id=None`` uses the session's active node.  Skips the
        synthetic root; carries system prompts from the session's
        ``GenerationConfig.system_prompt`` as the top-level
        ``system_prompt`` field rather than emitting a separate
        ``role: system`` turn (the YAML schema is flatter that way).
        """
        target = node_id if node_id is not None else session.tree.active_node_id
        path = session.tree.path_to(target)

        # Drop the synthetic root from the user-visible path; its empty
        # text would render as a no-op system turn.
        turns: list[Turn] = []
        for node in path:
            if node.id == session.tree.root_id:
                continue
            turn = Turn(
                role=node.role,
                text=node.text or "",
                speaker=node.role_label,
                thinking=node.thinking_text,
                recipe=node.recipe,
                readings=dict(node.aggregate_readings or {}),
            )
            turns.append(turn)

        probes = [
            ProbeRef(name=name, sha256=digest)
            for name, digest in session.probe_hashes().items()
        ]

        return cls(
            model_id=getattr(session, "model_id", None)
                or getattr(session.tree, "model_id", None),
            system_prompt=getattr(session.config, "system_prompt", None),
            probes=probes,
            turns=turns,
            cast=dict(getattr(session.tree, "cast", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "saklas_transcript": SAKLAS_TRANSCRIPT_VERSION,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
        }
        if self.cast:
            out["cast"] = {
                label: m.to_dict() for label, m in self.cast.items()
            }
        out["probes"] = [
            {"name": p.name, "sha256": p.sha256} for p in self.probes
        ]
        out["turns"] = [t.to_dict() for t in self.turns]
        return out

    def to_yaml(self) -> str:
        """Render to YAML (pyyaml is a hard saklas dependency)."""
        import yaml

        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, default_flow_style=False,
        )

    @classmethod
    def from_yaml(cls, text: str) -> "Transcript":
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise TranscriptFormatError(
                "pyyaml required to load transcripts (install with "
                "`pip install pyyaml`)"
            ) from e
        try:
            data = yaml.safe_load(text)
        except Exception as e:
            raise TranscriptFormatError(f"yaml parse error: {e}") from e
        if not isinstance(data, dict):
            raise TranscriptFormatError(
                f"transcript root must be a mapping, got {type(data).__name__}"
            )
        version = data.get("saklas_transcript")
        if version != SAKLAS_TRANSCRIPT_VERSION:
            raise TranscriptFormatError(
                f"unsupported saklas_transcript version {version!r} "
                f"(this build requires {SAKLAS_TRANSCRIPT_VERSION})"
            )
        probes = [
            ProbeRef(name=str(p["name"]), sha256=str(p.get("sha256", "")))
            for p in (data.get("probes") or [])
            if isinstance(p, dict) and "name" in p
        ]
        turns = [Turn.from_dict(t) for t in (data.get("turns") or [])]
        cast = {
            str(label): CastMember.from_dict(dict(raw))
            for label, raw in (data.get("cast") or {}).items()
        }
        return cls(
            model_id=data.get("model_id"),
            system_prompt=data.get("system_prompt"),
            probes=probes,
            turns=turns,
            cast=cast,
        )

    def save(self, path: str | Path) -> None:
        """Atomic-write the transcript to ``path``."""
        from saklas.io.atomic import write_bytes_atomic
        write_bytes_atomic(Path(path), self.to_yaml().encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    # ------------------------------------------------------------------
    # Import into a live session
    # ------------------------------------------------------------------

    def import_into(
        self,
        session: Any,
        *,
        mode: ImportMode = "default",
        strict: bool = False,
    ) -> str:
        """Attach this transcript to ``session.tree`` under ``mode``.

        Returns the imported branch's leaf node id.  The branch's root
        node carries any guard notes (model / system-prompt / probe
        mismatches) on its ``notes`` field so the surfaces can display
        a banner.

        Raises :class:`TranscriptModelMismatch` on model mismatch under
        ``mode="merge"`` and :class:`TranscriptProbeDriftError` on any
        probe hash difference when ``strict=True``.
        """
        guard_notes = self._collect_guard_notes(session, mode=mode, strict=strict)

        # Merge the transcript's cast roster into the tree's: absent
        # labels land, matching members are a no-op, and a label the
        # live roster holds with a *different* member is left alone —
        # the session's standing roster wins, with a guard note so the
        # conflict is visible on the imported branch.
        cast_conflicts: list[str] = []
        for label, member in self.cast.items():
            existing = session.tree.cast.get(label)
            if existing is None:
                session.tree.set_cast_member(label, member)
            elif existing != member:
                cast_conflicts.append(label)
        if cast_conflicts:
            msg = (
                f"transcript cast differs from the session roster for: "
                f"{', '.join(sorted(cast_conflicts))}; keeping the "
                f"session's members"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            guard_notes.append(f"cast_conflict: {sorted(cast_conflicts)}")

        attach_parent = self._resolve_attach_parent(session, mode=mode)
        return self._attach_turns_under(
            session.tree, attach_parent, guard_notes=guard_notes,
        )

    # ------------------------------------------------------------------
    # Guard checks
    # ------------------------------------------------------------------

    def _collect_guard_notes(
        self,
        session: Any,
        *,
        mode: ImportMode,
        strict: bool,
    ) -> list[str]:
        notes: list[str] = []

        session_model = (
            getattr(session, "model_id", None)
            or getattr(session.tree, "model_id", None)
        )
        if self.model_id and session_model and self.model_id != session_model:
            msg = (
                f"transcript model {self.model_id!r} differs from session "
                f"model {session_model!r}"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            if mode == "merge":
                raise TranscriptModelMismatch(
                    msg + " — refusing to merge under semantic mismatch"
                )
            notes.append(f"model_mismatch: {msg}")

        session_sys = getattr(session.config, "system_prompt", None)
        if (
            self.system_prompt is not None
            and session_sys is not None
            and self.system_prompt != session_sys
        ):
            msg = "transcript system prompt differs from current session"
            warnings.warn(msg, UserWarning, stacklevel=3)
            notes.append(
                f"system_prompt_mismatch: original was {self.system_prompt!r}"
            )

        session_hashes = session.probe_hashes()
        drift: list[str] = []
        missing: list[str] = []
        for ref in self.probes:
            current = session_hashes.get(ref.name)
            if current is None:
                missing.append(ref.name)
                continue
            if ref.sha256 and current != ref.sha256:
                drift.append(ref.name)
        if missing:
            msg = (
                f"transcript references probes the session doesn't carry: "
                f"{', '.join(sorted(missing))}; readings recorded as-imported"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            notes.append(f"probes_missing: {sorted(missing)}")
        if drift:
            msg = (
                f"probe content drift between transcript and session: "
                f"{', '.join(sorted(drift))}"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            if strict:
                raise TranscriptProbeDriftError(msg)
            notes.append(f"probe_drift: {sorted(drift)}")
        return notes

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def _resolve_attach_parent(
        self, session: Any, *, mode: ImportMode,
    ) -> str:
        if mode == "default":
            return session.tree.root_id
        if mode == "here":
            return session.tree.active_node_id
        # mode == "merge" — ``ImportMode`` is a Literal of the three
        # values above, so type-narrowing covers the universe.
        return self._find_merge_anchor(session)

    def _find_merge_anchor(self, session: Any) -> str:
        """Walk the active path; return the deepest user-turn match.

        Match = same user-turn text at the same position.  Assistant
        outputs are advisory and may differ (seed × steering × model
        × probe-state — byte-equal is rare).  Falls back to the tree
        root when no user-turn prefix matches.
        """
        active_path = session.tree.active_path()
        active_users: list[tuple[str, str]] = [
            (node.id, node.text or "")
            for node in active_path
            if node.role == "user"
        ]
        transcript_users = [t.text for t in self.turns if t.role == "user"]

        anchor_id: str = session.tree.root_id
        for (node_id, active_text), tr_text in zip(
            active_users, transcript_users, strict=False,
        ):
            if active_text != tr_text:
                break
            anchor_id = node_id
        return anchor_id

    # ------------------------------------------------------------------
    # Tree attachment
    # ------------------------------------------------------------------

    def _attach_turns_under(
        self,
        tree: LoomTree,
        attach_parent: str,
        *,
        guard_notes: list[str],
    ) -> str:
        """Spawn nodes mirroring ``self.turns`` under ``attach_parent``.

        When ``attach_parent`` is the merge anchor, the matching user-
        turn prefix is already in the tree — we skip that many ``user``
        entries from ``self.turns`` plus the assistant turns interleaved
        between them, and attach only the tail.  Skipped prefix nodes
        are not re-attached anywhere: the matched user chain already
        lives *above* ``attach_parent`` (the anchor is the deepest
        matched user), so re-adding them under the anchor would spawn
        duplicates as children of it.  Assistant turns interleaved with
        matched-prefix users are dropped on the same principle — their
        existing siblings on the tree's path are advisory, byte-equal
        comparison is rare, but the structural location is "already in
        the path" so attaching a copy under the merge anchor lands at
        the wrong depth.
        """
        # Determine the prefix to skip when merging — count how many
        # leading user-turn entries on the transcript share text with
        # the active path under ``attach_parent``.
        skip_count = 0
        if attach_parent != tree.root_id:
            # Walk transcript user turns vs path from root to attach_parent.
            anchor_path = tree.path_to(attach_parent)
            anchor_users = [n for n in anchor_path if n.role == "user"]
            transcript_users = [t for t in self.turns if t.role == "user"]
            for path_node, t_user in zip(anchor_users, transcript_users, strict=False):
                if path_node.text != t_user.text:
                    break
                skip_count += 1

        current_parent = attach_parent
        leaf_id = attach_parent

        # Track which user-turn we're at so we know when ``skip_count``
        # is satisfied — only ``user`` turns count toward the skip, but
        # any non-user turns before the skip is satisfied are inside
        # the matched-prefix region and skipped too.
        users_seen = 0
        first_imported_id: str | None = None
        for turn in self.turns:
            if turn.role == "system":
                # Schemas with explicit system turns are rare; treat
                # them as user turns under the synthetic root to
                # avoid altering established system-prompt semantics.
                continue
            if turn.role == "user":
                users_seen += 1
                if users_seen <= skip_count:
                    # Matched-prefix user: already represented in the
                    # tree path from root to ``attach_parent``.  Skip
                    # without writing — re-attaching here would create
                    # a duplicate user node as a child of the anchor.
                    continue
            elif users_seen < skip_count:
                # Assistant interleaved inside the matched prefix
                # region — drop on the floor (see docstring).
                continue
            if turn.role == "user" and turn.recipe is None:
                # Committed user turn — the plain shape.
                new_id = tree.add_user_turn(
                    turn.text, parent_id=current_parent,
                    dedup_existing=False, role_label=turn.speaker,
                    thinking_text=turn.thinking,
                )
            else:
                # A generated turn re-attaches in its recorded seat
                # (provenance = recipe presence — a user-seat gen keeps
                # its recipe); a recipe-less assistant turn is an
                # authored one and lands the same way with recipe=None.
                new_id = tree.begin_assistant(
                    current_parent, recipe=turn.recipe,
                    role_label=turn.speaker, seat=turn.role,
                )
                tree.finalize_assistant(
                    new_id,
                    text=turn.text,
                    aggregate_readings=dict(turn.readings),
                    applied_steering=(turn.recipe.steering if turn.recipe else None),
                    finish_reason=None,
                    thinking_text=turn.thinking,
                )
            current_parent = new_id
            leaf_id = new_id
            if first_imported_id is None:
                first_imported_id = new_id

        # Stamp guard notes on the first imported node (or the leaf
        # when no fresh nodes were created — the merge-and-nothing-new
        # case).
        if guard_notes:
            target = first_imported_id or leaf_id
            tree.annotate(target, "\n".join(guard_notes))

        return leaf_id


__all__ = [
    "ImportMode",
    "ProbeRef",
    "SAKLAS_TRANSCRIPT_VERSION",
    "Transcript",
    "TranscriptError",
    "TranscriptFormatError",
    "TranscriptModelMismatch",
    "TranscriptProbeDriftError",
    "Turn",
]
