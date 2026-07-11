"""Scene rendering — the cast-model turn stitcher.

The dynamic-roles design (``docs/plans/dynamic-roles.md``) needs renders no
chat template can produce: arbitrary seat sequences (a/a/u/a/u/u), per-turn
cast labels, generation prompts on either seat.  Gemma and Mistral templates
hard-raise on non-alternating roles, so ``apply_chat_template`` cannot be the
renderer — and hand-maintaining a per-family turn grammar would drift against
live templates (the failure mode ``RoleTemplateDriftError`` guards on the
splice path).

The resolution is **template autopsy**: render small sentinel-content probe
conversations through the *real* chat template once per model, locate the
sentinels, and mechanically slice out the segments — prelude, per-seat turn
wrappers (open / label site / close), the system-block shape (real turn vs
gemma-style fold into the first turn), and the generation-prompt appendix.
A scene then renders as pure **stitching**: ``open(seat, label) + content +
close(seat)`` per turn, plus a trailing generation header for whichever seat
speaks next.  Labels are placed in headers we *construct*, so the
occurrence-matching collision class of ``_splice_occurrences`` (a cast label
equal to the other seat's standard label corrupting the splice) is
structurally impossible here.

``validate_turn_grammar`` is the load-bearing check: stitch canonical
alternating conversations and byte-compare against the template's own render.
A family that passes gets scene mode — and, because extraction renders
exactly such alternating conversations, routing extraction through the
stitcher is bit-identical to ``apply_chat_template`` (the baseline-match
contract holds with no manifold re-fit).  A family that fails falls back to
raw-marker rendering (``render_scene_raw``) with a drift warning.

Thinking is a per-turn optional input on any seat (a9 convention, 2026-07-10):
rendered through the family think delimiters when supplied.  History policy
follows the family template's own convention — when the template strips
thinking from prior turns (detected empirically by the autopsy), the stitcher
renders a turn's thinking only while that turn is the *last* before the
generation header ("lasts one turn"); non-strip families keep it everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from saklas.core.errors import SaklasError
from saklas.core.role_templates import (
    ROLE_HEADERS,
    USER_ROLE_HEADERS,
    RoleHeader,
    _render_label,
    _validate_role,
)

Seat = Literal["user", "assistant"]

# Sentinel contents for the autopsy probes.  Single alphanumeric "words" so
# any reasonable template passes them through verbatim (no whitespace for a
# ``| trim`` filter to eat, nothing a Jinja escape would rewrite).  A template
# that transforms them fails sentinel location and the family falls back —
# self-guarding, not silently wrong.
_S_SYS = "SKLSPROBESYS"
_S_U1 = "SKLSPROBEUONE"
_S_A1 = "SKLSPROBEAONE"
_S_U2 = "SKLSPROBEUTWO"
_S_THINK = "SKLSPROBETHK"
# Whitespace-padded sentinels for the content-trim probe: many templates
# apply ``| trim`` to message content (gemma, llama, qwen3.5) and the
# stitcher must reproduce it or diverge on whitespace-adorned content.
_S_PAD_U = "  SKLSPADU  "
_S_PAD_A = "  SKLSPADA  "
_S_PAD_S = "  SKLSPADS  "


class SceneRenderError(SaklasError, RuntimeError):
    """A scene could not be rendered (bad seat, missing grammar support)."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class SceneGrammarError(SceneRenderError):
    """Template autopsy or round-trip validation failed for this family.

    Raised when the chat template's rendered bytes cannot be sliced into a
    consistent turn grammar (unknown/label-free family, first-turn-special
    template, sentinel transformation) or when the stitched reconstruction
    diverges from the template's own render.  Callers fall back to
    :func:`render_scene_raw`.
    """

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


class SceneThinkingUnsupportedError(SaklasError, ValueError):
    """A turn carries ``thinking`` but the grammar has no think delimiters."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


@dataclass(frozen=True)
class SceneTurn:
    """One line of a scene: ``(seat, label, text, thinking)``.

    ``seat`` is the structural side rendered into the token stream (binary —
    that is what the models' templates know).  ``label`` is the cast label
    spliced into the constructed header (``None`` = the family's standard
    label for that seat).  ``thinking`` is an optional verbatim reasoning
    block rendered through the family think delimiters.
    """

    seat: Seat
    text: str
    label: str | None = None
    thinking: str | None = None


@dataclass(frozen=True)
class SeatWrapper:
    """The literal bytes a family wraps around one seat's turn.

    ``open_before + <label> + open_after`` is the full turn header (the
    ``open_after`` carries any post-header bytes beyond the registry's
    ``RoleHeader.after``, e.g. llama's ``\\n\\n``); ``close`` is the
    terminator.  ``label`` is the family's standard label for the seat.
    """

    open_before: str
    open_after: str
    close: str
    label: str

    def open(self, label: str | None = None) -> str:
        rendered = (
            self.label if label is None else _render_label(self.label, label)
        )
        return f"{self.open_before}{rendered}{self.open_after}"


@dataclass(frozen=True)
class SystemShape:
    """A real system turn: ``before + content + after``."""

    before: str
    after: str


@dataclass(frozen=True)
class TurnGrammar:
    """Per-model turn segments extracted from the live chat template."""

    model_type: str
    # Bytes before the first turn when no system prompt is given (BOS, any
    # template-default system block — captured verbatim).
    prelude: str
    user: SeatWrapper
    assistant: SeatWrapper
    # At most one of the two system shapes is set: a real system turn, or a
    # gemma-style fold (system content + ``system_fold_sep`` prepended to the
    # first turn's content, whatever its seat — convention 1).  Both ``None``
    # = the family refuses the system role outright (gemma-2); rendering a
    # scene with ``system=`` then raises, mirroring the template.
    system: SystemShape | None
    system_fold_sep: str | None

    @property
    def system_supported(self) -> bool:
        return self.system is not None or self.system_fold_sep is not None
    # Bytes ``add_generation_prompt`` appends beyond the assistant open
    # (thinking scaffolds and the like), probed under ``enable_thinking=
    # False`` when the template takes that kwarg (matching
    # ``build_chat_input``'s default).  ``gen_extra_thinking`` is the
    # ``enable_thinking=True`` variant — equal to ``gen_extra`` when the
    # template ignores the kwarg, ``None`` when the thinking-mode render
    # defeated the slicer (rendering with ``gen_thinking=True`` then
    # raises and the caller falls back to the template).  User-seat
    # generation prompts have no template analogue; they are the user
    # open verbatim.
    gen_extra: str
    gen_extra_thinking: str | None = None
    # Family think delimiters (caller-supplied; ``None`` = thinking input
    # unsupported for this model) + the empirically-probed history policy.
    think_open: str | None = None
    think_close: str | None = None
    strips_history_thinking: bool = False
    # Whether the template trims turn content / system content (probed with
    # whitespace-padded sentinels).  Trim applies to the turn *text* before
    # thinking/fold composition — gemma-3's fold keeps the system content
    # raw while trimming the user text, so trim-then-compose is the
    # template-faithful order.
    content_trim: bool = False
    system_trim: bool = False
    # Some templates transform the final assistant turn of a closed render
    # (qwen3 inserts an empty think scaffold into it).  When set, a scene
    # ending on an assistant seat with no generation prompt is not
    # stitchable — callers fall back to the template (such renders are
    # alternating-shaped in practice: capture triples, scoring).
    last_assistant_special: bool = False

    def seat(self, name: str) -> SeatWrapper:
        # ``str``, not ``Seat`` — turns arrive from wire/loom surfaces, so
        # the guard below is a real runtime check, not decoration.
        if name == "user":
            return self.user
        if name == "assistant":
            return self.assistant
        raise SceneRenderError(
            f"unknown seat {name!r}: a scene turn occupies 'user' or "
            f"'assistant' (system content is the leading stage direction, "
            f"not a seat)"
        )


def _find_once(rendered: str, needle: str, *, what: str) -> int:
    n = rendered.count(needle)
    if n != 1:
        raise SceneGrammarError(
            f"template autopsy: expected exactly one occurrence of {what} "
            f"({needle!r}), found {n} — the chat template transforms content "
            f"or repeats it; scene mode unavailable for this family"
        )
    return rendered.index(needle)


def _render_probe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    **kwargs: Any,
) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — any template failure = no scene mode
        raise SceneGrammarError(
            f"template autopsy: probe render failed: {exc}"
        ) from exc
    if not isinstance(rendered, str):
        raise SceneGrammarError(
            f"template autopsy: probe render returned "
            f"{type(rendered).__name__}, expected str"
        )
    return rendered


def _split_header(
    segment: str, header: RoleHeader, *, what: str
) -> tuple[str, str]:
    """Split ``segment`` at the start of ``header``'s standard pattern.

    Returns ``(before_header, header_to_end)``.  Uses the *last* occurrence —
    the header nearest the content — so a template-default system block that
    happens to contain role-header bytes lands in the preceding segment.
    """
    pattern = f"{header.before}{header.label}{header.after}"
    idx = segment.rfind(pattern)
    if idx < 0:
        raise SceneGrammarError(
            f"template autopsy: {what} header {pattern!r} not found in "
            f"rendered segment {segment!r}"
        )
    return segment[:idx], segment[idx:]


def _wrapper_from_open(
    open_full: str, header: RoleHeader, close: str
) -> SeatWrapper:
    # ``open_full`` starts with ``before + label`` by construction
    # (``_split_header`` located that exact pattern).
    return SeatWrapper(
        open_before=header.before,
        open_after=open_full[len(header.before) + len(header.label):],
        close=close,
        label=header.label,
    )


def extract_turn_grammar(
    tokenizer: Any,
    model_type: str,
    *,
    think_delimiters: tuple[str, str] | None = None,
) -> TurnGrammar:
    """Autopsy the live chat template into a :class:`TurnGrammar`.

    Renders sentinel-content probe conversations (all strictly alternating,
    so even alternation-enforcing templates cooperate) and slices the
    rendered strings into reusable segments.  Raises
    :class:`SceneGrammarError` when the family is label-free / unknown
    (mistral) or the template's shape defeats the slicer (first-turn-special
    wrappers, content transforms); callers fall back to
    :func:`render_scene_raw`.

    ``think_delimiters`` are the family's ``(open, close)`` reasoning-block
    markers (from the generation layer's detector); when supplied, an extra
    probe determines empirically whether the template strips thinking from
    history, and the grammar's history policy follows it (a9 convention 3).
    """
    asst_header = ROLE_HEADERS.get(model_type)
    user_header = USER_ROLE_HEADERS.get(model_type)
    if asst_header is None or user_header is None:
        raise SceneGrammarError(
            f"scene mode is not supported for model_type {model_type!r}: "
            f"family is label-free or not in the role-header registry"
        )

    base_msgs = [
        {"role": "user", "content": _S_U1},
        {"role": "assistant", "content": _S_A1},
        {"role": "user", "content": _S_U2},
    ]
    # When the template takes ``enable_thinking``, every probe renders with
    # it pinned False — matching ``build_chat_input``'s default path — and a
    # separate probe pair recovers the True-variant generation appendix.
    template_str = getattr(tokenizer, "chat_template", "") or ""
    takes_thinking_kwarg = "enable_thinking" in template_str
    render_kwargs: dict[str, Any] = (
        {"enable_thinking": False} if takes_thinking_kwarg else {}
    )
    r0 = _render_probe(
        tokenizer, base_msgs, add_generation_prompt=False, **render_kwargs
    )
    r1 = _render_probe(
        tokenizer, base_msgs, add_generation_prompt=True, **render_kwargs
    )
    try:
        rs: str | None = _render_probe(
            tokenizer,
            [{"role": "system", "content": _S_SYS}] + base_msgs[:2],
            add_generation_prompt=False,
            **render_kwargs,
        )
    except SceneGrammarError:
        # A family whose template refuses the system role outright (gemma-2)
        # keeps scene mode without system support — ``render_scene`` raises
        # on a ``system=`` argument, mirroring the template's own behavior.
        rs = None

    i_u1 = _find_once(r0, _S_U1, what="user sentinel 1")
    i_a1 = _find_once(r0, _S_A1, what="assistant sentinel")
    i_u2 = _find_once(r0, _S_U2, what="user sentinel 2")
    if not (i_u1 < i_a1 < i_u2):
        raise SceneGrammarError(
            "template autopsy: sentinel order diverges from message order"
        )

    # Slice the alternating render into prelude + wrappers.
    prelude, user_open = _split_header(
        r0[:i_u1], user_header, what="first user"
    )
    user_close, asst_open = _split_header(
        r0[i_u1 + len(_S_U1):i_a1], asst_header, what="assistant"
    )
    asst_close, steady_user_open = _split_header(
        r0[i_a1 + len(_S_A1):i_u2], user_header, what="steady-state user"
    )
    if steady_user_open != user_open:
        raise SceneGrammarError(
            f"template autopsy: first-turn user header {user_open!r} differs "
            f"from steady-state {steady_user_open!r}; first-turn-special "
            f"templates are not stitchable"
        )
    tail = r0[i_u2 + len(_S_U2):]
    if tail != user_close:
        raise SceneGrammarError(
            f"template autopsy: trailing user close {tail!r} differs from "
            f"mid-conversation close {user_close!r}"
        )

    # Generation prompt: whatever add_generation_prompt appends, split into
    # the assistant open (label-swappable) + a fixed appendix.
    if not r1.startswith(r0):
        raise SceneGrammarError(
            "template autopsy: the generation-prompt render is not an "
            "extension of the plain render"
        )
    gen_append = r1[len(r0):]
    if not gen_append.startswith(asst_open):
        raise SceneGrammarError(
            f"template autopsy: generation prompt {gen_append!r} does not "
            f"begin with the assistant open {asst_open!r}"
        )
    gen_extra = gen_append[len(asst_open):]

    # The ``enable_thinking=True`` generation appendix.  Sliced from its own
    # probe pair (thinking mode may reshape history too, so the False-mode
    # ``r0`` is not a valid prefix reference).  A slicing defeat degrades to
    # ``None`` — thinking-mode stitching unsupported, not scene mode lost.
    gen_extra_thinking: str | None = gen_extra
    if takes_thinking_kwarg:
        gen_extra_thinking = None
        try:
            r0t = _render_probe(
                tokenizer, base_msgs, add_generation_prompt=False,
                enable_thinking=True,
            )
            r1t = _render_probe(
                tokenizer, base_msgs, add_generation_prompt=True,
                enable_thinking=True,
            )
            # ``r0t == r0`` is required, not just prefix consistency: the
            # stitched history always uses the False-mode wrappers, so a
            # template that reshapes *history* under the flag cannot be
            # reproduced in thinking mode at all.
            if r0t == r0 and r1t.startswith(r0t):
                gen_append_t = r1t[len(r0t):]
                if gen_append_t.startswith(asst_open):
                    gen_extra_thinking = gen_append_t[len(asst_open):]
        except SceneGrammarError:
            pass

    # System shape: a real turn (sentinel precedes the first user header) or
    # a fold into the first turn's content (gemma).
    system: SystemShape | None = None
    system_fold_sep: str | None = None
    if rs is not None:
        _find_once(rs, _S_SYS, what="system sentinel")
        i_u1s = _find_once(rs, _S_U1, what="user sentinel (system probe)")
        sys_prefix, rest = _split_header(
            rs[:i_u1s], user_header, what="user (system probe)"
        )
        if not rest.startswith(user_open):
            raise SceneGrammarError(
                "template autopsy: the user header changes shape when a "
                "system prompt is present; not stitchable"
            )
        # ``after_open`` is what sits between the user open and the first
        # user content: empty for a real-system-turn family, ``system +
        # sep`` for a fold family (gemma).
        after_open = rest[len(user_open):]
        if _S_SYS in sys_prefix:
            if after_open:
                raise SceneGrammarError(
                    "template autopsy: system content appears both before "
                    "and inside the first turn; not stitchable"
                )
            j = sys_prefix.index(_S_SYS)
            system = SystemShape(
                before=sys_prefix[:j],
                after=sys_prefix[j + len(_S_SYS):],
            )
        elif after_open.startswith(_S_SYS) and sys_prefix == prelude:
            system_fold_sep = after_open[len(_S_SYS):]
        else:
            raise SceneGrammarError(
                "template autopsy: system content lands neither in a leading "
                "block nor at the head of the first turn; not stitchable"
            )

    # Content-trim probe: does the template ``| trim`` message content?
    # The padded turns sit in *history* position (a plain user turn closes
    # the probe) so a last-turn-special transform can't contaminate the
    # trim verdict.
    r_pad = _render_probe(
        tokenizer,
        [
            {"role": "user", "content": _S_PAD_U},
            {"role": "assistant", "content": _S_PAD_A},
            {"role": "user", "content": _S_U2},
        ],
        add_generation_prompt=False,
        **render_kwargs,
    )
    user_trim = _S_PAD_U not in r_pad
    asst_trim = _S_PAD_A not in r_pad
    if user_trim != asst_trim:
        raise SceneGrammarError(
            "template autopsy: the template trims one seat's content but "
            "not the other's; not stitchable"
        )
    if user_trim and _S_PAD_U.strip() not in r_pad:
        raise SceneGrammarError(
            "template autopsy: content transformation is not a plain trim; "
            "not stitchable"
        )
    content_trim = user_trim
    system_trim = False
    if system is not None:
        rs_pad = _render_probe(
            tokenizer,
            [
                {"role": "system", "content": _S_PAD_S},
                {"role": "user", "content": _S_U1},
            ],
            add_generation_prompt=False,
            **render_kwargs,
        )
        system_trim = _S_PAD_S not in rs_pad
        if system_trim and _S_PAD_S.strip() not in rs_pad:
            raise SceneGrammarError(
                "template autopsy: system-content transformation is not a "
                "plain trim; not stitchable"
            )

    # Last-assistant-turn probe: a closed render ending on an assistant
    # turn may be transformed (qwen3's empty-think insert).  Compare the
    # template's own [user, assistant] render against the stitched
    # reconstruction from the extracted wrappers.
    r_last = _render_probe(
        tokenizer,
        [
            {"role": "user", "content": _S_U1},
            {"role": "assistant", "content": _S_A1},
        ],
        add_generation_prompt=False,
        **render_kwargs,
    )
    last_assistant_special = r_last != (
        f"{prelude}{user_open}{_S_U1}{user_close}"
        f"{asst_open}{_S_A1}{asst_close}"
    )

    think_open: str | None = None
    think_close: str | None = None
    strips = False
    if think_delimiters is not None:
        think_open, think_close = think_delimiters
        try:
            rt = _render_probe(
                tokenizer,
                [
                    {"role": "user", "content": _S_U1},
                    {
                        "role": "assistant",
                        "content": f"{think_open}{_S_THINK}{think_close}{_S_A1}",
                    },
                    {"role": "user", "content": _S_U2},
                ],
                add_generation_prompt=True,
                **render_kwargs,
            )
            strips = _S_THINK not in rt
        except SceneGrammarError:
            # A template that chokes on embedded think markers loses the
            # thinking *feature*, not scene mode.
            think_open = think_close = None

    return TurnGrammar(
        model_type=model_type,
        prelude=prelude,
        user=_wrapper_from_open(user_open, user_header, user_close),
        assistant=_wrapper_from_open(asst_open, asst_header, asst_close),
        system=system,
        system_fold_sep=system_fold_sep,
        gen_extra=gen_extra,
        gen_extra_thinking=gen_extra_thinking,
        think_open=think_open,
        think_close=think_close,
        strips_history_thinking=strips,
        content_trim=content_trim,
        system_trim=system_trim,
        last_assistant_special=last_assistant_special,
    )


def render_scene(
    grammar: TurnGrammar,
    turns: list[SceneTurn],
    *,
    system: str | None = None,
    gen_seat: Seat | None = None,
    gen_label: str | None = None,
    gen_thinking: bool = False,
) -> str:
    """Stitch a scene into the exact string the model should ingest.

    ``turns`` may occupy seats in any order — alternation is not required,
    which is the point.  ``gen_seat`` appends a trailing generation header
    for the turn about to be generated (``gen_label`` is its cast label);
    ``None`` renders a closed transcript (the extraction/capture shape).

    Tokenize the result with ``add_special_tokens=False`` — the prelude
    already carries whatever BOS the template emits (the same contract as
    the splice path).
    """
    for turn in turns:
        if turn.label is not None:
            _validate_role(turn.label)
        if turn.thinking is not None and grammar.think_open is None:
            raise SceneThinkingUnsupportedError(
                f"model_type {grammar.model_type!r} has no thinking "
                f"delimiters; a scene turn cannot carry a thinking block"
            )
    if gen_label is not None:
        _validate_role(gen_label)

    parts: list[str] = []
    fold: str | None = None
    if system is not None and grammar.system is not None:
        sys_content = system.strip() if grammar.system_trim else system
        parts.append(
            f"{grammar.system.before}{sys_content}{grammar.system.after}"
        )
    elif system is not None and grammar.system_fold_sep is not None:
        # Fold family (convention 1): system content prepends into the first
        # turn, whatever its seat.
        parts.append(grammar.prelude)
        fold = f"{system}{grammar.system_fold_sep}"
    elif system is not None:
        raise SceneRenderError(
            f"model_type {grammar.model_type!r} does not support system "
            f"prompts (the chat template refuses the system role)"
        )
    else:
        parts.append(grammar.prelude)

    last = len(turns) - 1
    if (
        turns
        and gen_seat is None
        and turns[-1].seat == "assistant"
        and grammar.last_assistant_special
    ):
        raise SceneRenderError(
            f"model_type {grammar.model_type!r} transforms the final "
            f"assistant turn of a closed render (e.g. qwen3's empty-think "
            f"insert); this shape must render through the chat template"
        )
    for i, turn in enumerate(turns):
        wrapper = grammar.seat(turn.seat)
        # Trim applies to the turn *text*, before thinking/fold composition
        # (gemma-3 folds the raw system content around trimmed text).
        content = turn.text.strip() if grammar.content_trim else turn.text
        if turn.thinking is not None:
            # History policy follows the family template (a9 convention 3):
            # strip families keep a thinking block only while its turn is
            # the last before the generation header — "lasts one turn".
            if not grammar.strips_history_thinking or i == last:
                content = (
                    f"{grammar.think_open}{turn.thinking}"
                    f"{grammar.think_close}{content}"
                )
        if i == 0 and fold is not None:
            content = f"{fold}{content}"
        parts.append(f"{wrapper.open(turn.label)}{content}{wrapper.close}")

    if gen_seat is not None:
        wrapper = grammar.seat(gen_seat)
        parts.append(wrapper.open(gen_label))
        if gen_seat == "assistant":
            if gen_thinking:
                if grammar.gen_extra_thinking is None:
                    raise SceneRenderError(
                        f"model_type {grammar.model_type!r}: the thinking-"
                        f"mode generation appendix defeated the template "
                        f"autopsy; render with gen_thinking=False or fall "
                        f"back to the chat template"
                    )
                parts.append(grammar.gen_extra_thinking)
            else:
                parts.append(grammar.gen_extra)
    return "".join(parts)


def render_scene_raw(
    turns: list[SceneTurn],
    *,
    system: str | None = None,
    gen_seat: Seat | None = None,
    gen_label: str | None = None,
) -> str:
    """Marker-mode fallback: the flat buffer generalized from one voice to N.

    The base-model path and the escape hatch for families whose template
    defeats the autopsy (mistral's positional markers).  ``label: text``
    lines, blank-line separated; thinking blocks are not rendered (a raw
    surface has no delimiters to strip by, so "lasts one turn" has no
    referent).
    """

    def display(label: str | None, seat: Seat) -> str:
        if label is not None:
            _validate_role(label)
            return _render_label(seat, label).capitalize()
        return seat.capitalize()

    parts = [] if system is None else [system]
    parts.extend(
        f"{display(t.label, t.seat)}: {t.text}" for t in turns
    )
    tail = ""
    if gen_seat is not None:
        tail = f"\n\n{display(gen_label, gen_seat)}:"
    return "\n\n".join(parts) + tail


def validate_turn_grammar(
    grammar: TurnGrammar,
    tokenizer: Any,
    *,
    system: str = "Answer briefly.",
) -> None:
    """Round-trip check: the stitcher must reproduce the template's bytes.

    Stitches canonical strictly-alternating conversations (standard labels)
    and byte-compares against ``apply_chat_template`` — plain, with a
    generation prompt, and with a system prompt.  Raises
    :class:`SceneGrammarError` at the first divergence.  This is the
    load-bearing gate: a passing family's extraction renders are bit-identical
    through the stitcher (the baseline-match contract), and a template
    revision flips the family to the raw fallback instead of corrupting
    silently.
    """
    messages = [
        {"role": "user", "content": "alpha beta"},
        {"role": "assistant", "content": "gamma delta"},
        {"role": "user", "content": "epsilon"},
    ]
    turns = [
        SceneTurn(seat=m["role"], text=m["content"])  # type: ignore[arg-type]
        for m in messages
    ]
    padded = [
        {"role": "user", "content": "  alpha beta  "},
        {"role": "assistant", "content": "  gamma  delta  "},
        {"role": "user", "content": "epsilon"},
    ]
    padded_turns = [
        SceneTurn(seat=m["role"], text=m["content"])  # type: ignore[arg-type]
        for m in padded
    ]
    cases: list[tuple[str, list[dict[str, str]], list[SceneTurn], str | None, Seat | None]] = [
        ("plain", messages, turns, None, None),
        ("generation prompt", messages, turns, None, "assistant"),
        # Whitespace-padded content — proves the trim probe's verdict, so a
        # template with a content transform the stitcher doesn't model can
        # never pass silently.
        ("padded content", padded, padded_turns, None, None),
    ]
    if not grammar.last_assistant_special:
        # A closed render ending on an assistant turn (the capture-triple /
        # scoring shape) — proves the last-assistant probe's verdict.
        cases.append((
            "closed on assistant",
            messages[:2],
            turns[:2],
            None,
            None,
        ))
    if grammar.system_supported:
        cases.append((
            "system prompt",
            [{"role": "system", "content": system}] + messages,
            turns,
            system,
            None,
        ))
        cases.append((
            "padded system",
            [{"role": "system", "content": f"  {system}  "}] + padded,
            padded_turns,
            f"  {system}  ",
            None,
        ))
    template_str = getattr(tokenizer, "chat_template", "") or ""
    takes_thinking_kwarg = "enable_thinking" in template_str
    render_kwargs: dict[str, Any] = (
        {"enable_thinking": False} if takes_thinking_kwarg else {}
    )
    for name, msgs, scene_turns, sys_prompt, gen_seat in cases:
        expected = _render_probe(
            tokenizer, msgs, add_generation_prompt=gen_seat is not None,
            **render_kwargs,
        )
        got = render_scene(
            grammar, scene_turns, system=sys_prompt, gen_seat=gen_seat
        )
        if got != expected:
            div = next(
                (k for k, (a, b) in enumerate(zip(expected, got)) if a != b),
                min(len(expected), len(got)),
            )
            raise SceneGrammarError(
                f"round-trip validation failed ({name}) for model_type "
                f"{grammar.model_type!r} at byte {div}: template "
                f"{expected[div:div + 40]!r} vs stitched {got[div:div + 40]!r}"
            )
    # Thinking-mode generation prompt, when the template takes the kwarg
    # and the autopsy recovered the variant.
    if takes_thinking_kwarg and grammar.gen_extra_thinking is not None:
        expected = _render_probe(
            tokenizer, messages, add_generation_prompt=True,
            enable_thinking=True,
        )
        got = render_scene(
            grammar, turns, gen_seat="assistant", gen_thinking=True
        )
        if got != expected:
            raise SceneGrammarError(
                f"round-trip validation failed (thinking generation prompt) "
                f"for model_type {grammar.model_type!r}: template "
                f"{expected[-60:]!r} vs stitched {got[-60:]!r}"
            )
