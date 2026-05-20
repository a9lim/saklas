"""Slash-command registry for the TUI.

Replaces the if/elif dispatch in ``SaklasApp._handle_command`` with a
table of :class:`SlashCommand` entries. Each command names its handler
(an unbound method on :class:`SaklasApp`), the usage string the
dispatcher prints on a shape mismatch, the min/max whitespace-token
count for validation, and an ``interrupts`` flag for handlers that need
to defer until the current generation finishes.

Handler signature contract: ``handler(app: SaklasApp, raw_args: str) ->
None``. ``raw_args`` is everything after ``<command_name> `` (possibly
empty), preserving quotes and embedded whitespace. Handlers that want
token-style parsing call ``raw_args.split()`` or ``shlex.split``
themselves; handlers that consume the rest of the line as a single
argument (``/sys <prompt>``, ``/save <name>``, ``/probe <pos> . <neg>``)
get the unsplit remainder.

Keeping the handlers as methods on :class:`SaklasApp` (rather than free
functions in this module) avoids the cycle that would otherwise form —
the registry references the unbound methods, and the dispatcher binds
them to ``self`` at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from saklas.tui.app import SaklasApp


@dataclass(frozen=True)
class SlashCommand:
    """Single entry in the slash-command dispatch table.

    Attributes
    ----------
    name:
        The slash-prefixed command (``"/steer"``).
    handler:
        Unbound method on :class:`SaklasApp` taking ``(app, raw_args)``.
    usage:
        Multi-line usage string the dispatcher prints when the
        whitespace-token count of ``raw_args`` falls outside
        ``[min_args, max_args]``.
    min_args:
        Lower bound on whitespace-token count. ``0`` means the command
        accepts no-arg form (typically "show current value").
    max_args:
        Upper bound; ``None`` = unbounded. Use ``None`` for handlers
        that consume the full remainder as one logical argument
        (``/probe <pos> . <neg>``, ``/sys <prompt>``).
    interrupts:
        If ``True`` and a generation is in flight (or earlier items
        sit on the pending queue), the dispatcher enqueues a
        :class:`PendingItem` with ``kind=pending_kind`` and lets the
        current gen run to completion.  The handler fires from
        ``_dispatch_pending_action`` when the queue head is drained
        on the next ``done``.  Non-``interrupts`` slash commands —
        ``/temp``, ``/seed``, ``/sys``, …  — run immediately even
        mid-gen because they don't conflict with the in-flight
        stream.
    pending_kind:
        ``kind`` tag for the queued :class:`PendingItem`.  Required
        when ``interrupts=True``.
    """

    name: str
    handler: Callable[["SaklasApp", str], None]
    usage: str
    min_args: int
    max_args: int | None
    interrupts: bool = False
    pending_kind: str | None = None


def _tok_count(raw_args: str) -> int:
    return len(raw_args.split()) if raw_args.strip() else 0


def _build_registry() -> dict[str, SlashCommand]:
    """Construct the canonical command table.

    Built lazily inside a function so :class:`SaklasApp` is fully
    defined by import time — the registry holds references to its
    unbound methods.
    """
    from saklas.tui.app import SaklasApp

    def _regen(app: "SaklasApp", raw: str) -> None:
        # Phase 4: ``/regen [N]`` — default 1, fan out N siblings under the
        # active assistant's user-parent.  N=1 keeps today's behavior bit-
        # identical so the registry interrupts/pending-action wiring is
        # unchanged for the common path.
        #
        # Phase 5 extension: trailing ``<mode>`` token (``unsteered`` /
        # ``inverted`` / ``reseed`` / ``cool`` / ``hot``, or ``custom:
        # <steering expression>``) flips the regen over to
        # ``session.regen_with_modifier``.  Custom modes parse the
        # expression into a Recipe partial up front so the worker can
        # route through ``compose_modifier(Recipe)`` directly.
        from saklas import Recipe

        raw = (raw or "").strip()
        tokens = raw.split() if raw else []
        n = 1
        mode: "str | Recipe | None" = None
        if tokens:
            try:
                n = max(1, int(tokens[0]))
            except ValueError:
                app._chat_panel.add_system_message(f"/regen: bad N '{tokens[0]}'")
                return
            if len(tokens) > 1:
                mode_str = " ".join(tokens[1:])
                if mode_str.lower().startswith("custom:"):
                    recipe = app._parse_custom_auto_regen(mode_str)
                    if recipe is None:
                        return  # parse error already posted to chat
                    mode = recipe
                else:
                    mode = mode_str
        if mode is not None:
            app._dispatch_loom_regen(n, mode=mode)
            return
        if n == 1:
            app.action_regenerate()
            return
        app._dispatch_loom_regen(n)

    def _quit(app: "SaklasApp", _raw: str) -> None:
        app.exit()

    def _clear(app: "SaklasApp", _raw: str) -> None:
        app._do_clear()

    def _rewind(app: "SaklasApp", _raw: str) -> None:
        app._do_rewind()

    def _model(app: "SaklasApp", _raw: str) -> None:
        app._handle_model_info()

    cmds: list[SlashCommand] = [
        SlashCommand(
            name="/steer",
            handler=SaklasApp._handle_steer,
            usage=(
                "Usage: /steer <expression>\n"
                "  e.g. /steer 0.5 honest\n"
                "       /steer 0.3 warm@after\n"
                "       /steer 0.5 honest:sae\n"
                "       /steer alice/                (bulk; default-off)\n"
                "  For a new bipolar extraction, use /extract <pos> <neg>."
            ),
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/alpha",
            handler=SaklasApp._handle_alpha,
            usage="Usage: /alpha <value> <name>",
            min_args=1,
            max_args=None,  # _handle_alpha uses shlex.split for quoted names
        ),
        SlashCommand(
            name="/unsteer",
            handler=SaklasApp._handle_unsteer,
            usage="Usage: /unsteer <name>  |  /unsteer <ns>/  (bulk)",
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/probe",
            handler=SaklasApp._handle_probe,
            usage=(
                "Usage: /probe <concept>\n"
                "       /probe <pos> . <neg>\n"
                "       /probe <ns>/         (bulk add namespace)"
            ),
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/unprobe",
            handler=SaklasApp._handle_unprobe,
            usage="Usage: /unprobe <name>  |  /unprobe <ns>/  (bulk)",
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/extract",
            handler=SaklasApp._handle_extract_only,
            usage=(
                "Usage: /extract <concept>\n"
                "       /extract <pos> . <neg>"
            ),
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/seed",
            handler=SaklasApp._handle_seed,
            usage="Usage: /seed [n|clear]",
            min_args=0,
            max_args=1,
        ),
        SlashCommand(
            name="/save",
            handler=SaklasApp._handle_save,
            usage="Usage: /save <name>",
            min_args=1,
            max_args=1,
        ),
        SlashCommand(
            name="/load",
            handler=SaklasApp._handle_load,
            usage="Usage: /load <name>",
            min_args=1,
            max_args=1,
        ),
        SlashCommand(
            name="/export",
            handler=SaklasApp._handle_export,
            usage="Usage: /export <path>",
            min_args=1,
            max_args=None,  # paths may contain whitespace
        ),
        SlashCommand(
            name="/regen",
            handler=_regen,
            usage="Usage: /regen [N] [mode]   (mode: unsteered|inverted|reseed|cool|hot)",
            min_args=0,
            max_args=None,
        ),
        SlashCommand(
            name="/model",
            handler=_model,
            usage="Usage: /model",
            min_args=0,
            max_args=0,
        ),
        SlashCommand(
            name="/clear",
            handler=_clear,
            usage="Usage: /clear",
            min_args=0,
            max_args=0,
            interrupts=True,
            pending_kind="clear",
        ),
        SlashCommand(
            name="/rewind",
            handler=_rewind,
            usage="Usage: /rewind",
            min_args=0,
            max_args=0,
            interrupts=True,
            pending_kind="rewind",
        ),
        SlashCommand(
            name="/sys",
            handler=SaklasApp._handle_sys,
            usage="Usage: /sys [prompt]",
            min_args=0,
            max_args=None,  # prompts contain spaces
        ),
        SlashCommand(
            name="/system",
            handler=SaklasApp._handle_sys,
            usage="Usage: /system [prompt]",
            min_args=0,
            max_args=None,
        ),
        SlashCommand(
            name="/temp",
            handler=SaklasApp._handle_temp,
            usage="Usage: /temp [value]",
            min_args=0,
            max_args=1,
        ),
        SlashCommand(
            name="/top-p",
            handler=SaklasApp._handle_top_p,
            usage="Usage: /top-p [value]",
            min_args=0,
            max_args=1,
        ),
        SlashCommand(
            name="/max",
            handler=SaklasApp._handle_max,
            usage="Usage: /max [value]",
            min_args=0,
            max_args=1,
        ),
        SlashCommand(
            name="/exit",
            handler=_quit,
            usage="Usage: /exit",
            min_args=0,
            max_args=0,
            interrupts=True,
            pending_kind="quit",
        ),
        SlashCommand(
            name="/quit",
            handler=_quit,
            usage="Usage: /quit",
            min_args=0,
            max_args=0,
            interrupts=True,
            pending_kind="quit",
        ),
        SlashCommand(
            name="/compare",
            handler=SaklasApp._handle_compare,
            usage="Usage: /compare <name> [other_name]",
            min_args=1,
            max_args=2,
        ),
        SlashCommand(
            name="/help",
            handler=SaklasApp._handle_help,
            usage="Usage: /help",
            min_args=0,
            max_args=0,
        ),
        SlashCommand(
            name="/commit",
            handler=SaklasApp._handle_commit,
            # ``min_args=1, max_args=None`` so the rest of the line is
            # one logical text argument — same shape as ``/sys`` and
            # ``/note``.  Slash-dispatch passes the full remainder to
            # the handler, preserving internal whitespace.
            usage="Usage: /commit <text>",
            min_args=1,
            max_args=None,
        ),
        # --- Loom (phase 4) ---
        SlashCommand(
            name="/tree",
            handler=SaklasApp._handle_tree,
            usage="Usage: /tree",
            min_args=0,
            max_args=0,
        ),
        SlashCommand(
            name="/nav",
            handler=SaklasApp._handle_nav,
            usage="Usage: /nav <id-prefix>",
            min_args=1,
            max_args=1,
        ),
        SlashCommand(
            name="/edit",
            handler=SaklasApp._handle_edit,
            usage="Usage: /edit <text...>",
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/branch",
            handler=SaklasApp._handle_branch,
            usage="Usage: /branch [text...]",
            min_args=0,
            max_args=None,
        ),
        SlashCommand(
            name="/del",
            handler=SaklasApp._handle_del,
            usage="Usage: /del [yes]",
            min_args=0,
            max_args=1,
        ),
        SlashCommand(
            name="/star",
            handler=SaklasApp._handle_star,
            usage="Usage: /star",
            min_args=0,
            max_args=0,
        ),
        SlashCommand(
            name="/note",
            handler=SaklasApp._handle_note,
            usage="Usage: /note <text>",
            min_args=0,
            max_args=None,
        ),
        SlashCommand(
            name="/path",
            handler=SaklasApp._handle_path,
            usage="Usage: /path",
            min_args=0,
            max_args=0,
        ),
        SlashCommand(
            name="/fan",
            handler=SaklasApp._handle_fan,
            usage=(
                "Usage: /fan <vector> <alphas>\n"
                "  alphas: '0.0, 0.3, 0.7'  ·  linspace(-1, 1, 5)  ·  0.0:1.0:0.25"
            ),
            min_args=2,
            max_args=None,
        ),
        SlashCommand(
            name="/prune",
            handler=SaklasApp._handle_prune,
            usage="Usage: /prune <filter-expr>  (phase 5 evaluates; phase 4 stashes)",
            min_args=0,
            max_args=None,
        ),
        SlashCommand(
            name="/auto-regen",
            handler=SaklasApp._handle_auto_regen,
            usage="Usage: /auto-regen [unsteered|inverted|reseed|cool|hot|<expr>]",
            min_args=0,
            max_args=None,
        ),
        SlashCommand(
            name="/diff",
            handler=SaklasApp._handle_diff,
            usage=(
                "Usage: /diff <id1> <id2> [--full]\n"
                "       /diff --siblings"
            ),
            min_args=1,
            max_args=None,
        ),
    ]
    return {c.name: c for c in cmds}


# Built lazily so importing this module doesn't trigger ``saklas.tui.app``
# during ``saklas.tui.app``'s own import.
_REGISTRY: dict[str, SlashCommand] | None = None


def get_registry() -> dict[str, SlashCommand]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def dispatch(
    app: "SaklasApp", text: str, *, replace_slot: int | None = None,
) -> None:
    """Route a raw ``/...`` line through the registry.

    * unknown command → ``"Unknown command: ... Type /help for commands."``
    * arg shape mismatch → ``cmd.usage``
    * ``interrupts=True`` while ``app._is_busy`` → enqueue a
      :class:`PendingItem` so the in-flight gen runs to completion;
      the handler fires once the queue head is drained.
    * otherwise → ``cmd.handler(app, raw_args)``

    ``replace_slot`` is forwarded to the queue when this dispatch
    came from a pulled-and-re-edited pending row, so the new item
    replaces its slot rather than sliding to the queue tail.
    """
    from saklas.tui.chat_panel import PendingItem

    text = text.strip()
    parts = text.split(maxsplit=1)
    if not parts:
        return
    name = parts[0].lower()
    raw_args = parts[1] if len(parts) > 1 else ""

    cmd = get_registry().get(name)
    if cmd is None:
        app._chat_panel.add_system_message(
            f"Unknown command: {name}. Type /help for commands."
        )
        return

    n = _tok_count(raw_args)
    if n < cmd.min_args or (cmd.max_args is not None and n > cmd.max_args):
        app._chat_panel.add_system_message(cmd.usage)
        return

    if cmd.interrupts and app._is_busy:
        # Enqueue a PendingItem; ``_poll_generation`` drains the
        # queue head on each ``("done",)`` sentinel and calls
        # ``_dispatch_pending_action``.  Carry the full slash text
        # so the user can pull and edit the queued item via ↑.
        app._enqueue_pending(
            PendingItem(cmd.pending_kind or cmd.name.lstrip("/"), text),
            replace_slot=replace_slot,
        )
        return

    cmd.handler(app, raw_args)
