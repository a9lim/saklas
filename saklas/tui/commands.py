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
        If ``True`` and a generation is in flight, the dispatcher
        stashes ``(pending_kind, raw_args)`` on ``app._pending_action``
        and calls ``session.stop()`` instead of running the handler
        immediately. The handler runs from ``_dispatch_pending_action``
        once ``("done",)`` lands on the UI queue.
    pending_kind:
        Tag for ``_pending_action``. Required when ``interrupts=True``.
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

    def _regen(app: "SaklasApp", _raw: str) -> None:
        app.action_regenerate()

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
            usage="Usage: /unsteer <name>",
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/probe",
            handler=SaklasApp._handle_probe,
            usage=(
                "Usage: /probe <concept>\n"
                "       /probe <pos> . <neg>"
            ),
            min_args=1,
            max_args=None,
        ),
        SlashCommand(
            name="/unprobe",
            handler=SaklasApp._handle_unprobe,
            usage="Usage: /unprobe <name>",
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
            usage="Usage: /regen",
            min_args=0,
            max_args=0,
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


def dispatch(app: "SaklasApp", text: str) -> None:
    """Route a raw ``/...`` line through the registry.

    Mirrors the old if/elif dispatch one-for-one:

    * unknown command → ``"Unknown command: ... Type /help for commands."``
    * arg shape mismatch → ``cmd.usage``
    * ``interrupts=True`` while ``session.is_generating`` → stash on
      ``_pending_action`` + ``session.stop()``
    * otherwise → ``cmd.handler(app, raw_args)``
    """
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

    if cmd.interrupts and app._session.is_generating:
        # Stash the pending kind; ``_poll_generation`` consumes the
        # ``("done",)`` sentinel and calls ``_dispatch_pending_action``.
        # Pending kinds for /clear, /rewind, /quit don't carry args; the
        # tuple shape stays a 1-element tuple to match the existing
        # dispatch contract.
        app._pending_action = (cmd.pending_kind,)
        app._session.stop()
        return

    cmd.handler(app, raw_args)
