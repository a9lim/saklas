"""Phase 1 logit-pass tests — engine-side logprob capture, top-K alts,
``LoomNode.mean_logprob`` aggregation.

Mostly CPU-only structural tests since the new surfaces are typed
dataclasses + composition logic that doesn't need a model on disk. The
end-to-end "captured logprob matches log_softmax" check would require a
real model load and lives in ``test_session.py`` (GPU-gated) once it's
sensible to add there.
"""
from __future__ import annotations

import pytest

from saklas import TokenAlt
from saklas.core.results import GenerationResult, TokenEvent
from saklas.core.sampling import SamplingConfig
from saklas.core.loom import LoomNode, LoomTree


# ---------------------------------------------------------------------------
# SamplingConfig.return_top_k
# ---------------------------------------------------------------------------


class TestReturnTopKConfig:
    def test_default_zero(self):
        sc = SamplingConfig()
        assert sc.return_top_k == 0

    def test_explicit_value(self):
        sc = SamplingConfig(return_top_k=8)
        assert sc.return_top_k == 8

    def test_clamp_negative_to_zero(self):
        sc = SamplingConfig(return_top_k=-3)
        assert sc.return_top_k == 0

    def test_clamp_above_256(self):
        sc = SamplingConfig(return_top_k=999)
        assert sc.return_top_k == 256

    def test_boundary_256_unchanged(self):
        sc = SamplingConfig(return_top_k=256)
        assert sc.return_top_k == 256

    def test_merged_with_override(self):
        base = SamplingConfig(temperature=0.5)
        override = SamplingConfig(return_top_k=4)
        merged = base.merged_with(override)
        assert merged.return_top_k == 4
        assert merged.temperature == 0.5  # unrelated field preserved

    def test_merged_with_default_is_noop(self):
        """``return_top_k=0`` is the SamplingConfig default — merged_with
        treats it as "no override," same discipline as the other fields
        that default to ``None`` / neutral sentinels."""
        base = SamplingConfig(return_top_k=8)
        override = SamplingConfig()  # return_top_k=0 (the default)
        merged = base.merged_with(override)
        assert merged.return_top_k == 8


# ---------------------------------------------------------------------------
# TokenAlt dataclass shape
# ---------------------------------------------------------------------------


class TestTokenAlt:
    def test_fields(self):
        alt = TokenAlt(id=42, text=" hello", logprob=-2.3)
        assert alt.id == 42
        assert alt.text == " hello"
        assert alt.logprob == pytest.approx(-2.3)

    def test_frozen(self):
        alt = TokenAlt(id=1, text="a", logprob=-1.0)
        with pytest.raises((AttributeError, Exception)):
            alt.id = 2  # type: ignore[misc]

    def test_equality(self):
        a = TokenAlt(id=7, text="x", logprob=-0.5)
        b = TokenAlt(id=7, text="x", logprob=-0.5)
        assert a == b
        assert hash(a) == hash(b)

    def test_reexported_from_top_level(self):
        """``TokenAlt`` is re-exported from ``saklas`` for the same
        reason ``TokenEvent`` / ``SamplingConfig`` are — library users
        shouldn't have to reach into private submodule paths."""
        import saklas
        assert saklas.TokenAlt is TokenAlt


# ---------------------------------------------------------------------------
# TokenEvent — new top_alts field
# ---------------------------------------------------------------------------


class TestTokenEventTopAlts:
    def test_default_none(self):
        ev = TokenEvent(text="x", token_id=0, index=0)
        assert ev.top_alts is None

    def test_carries_alts(self):
        alts = [
            TokenAlt(id=1, text=" hello", logprob=-1.2),
            TokenAlt(id=2, text=" hi", logprob=-2.4),
        ]
        ev = TokenEvent(text=" hello", token_id=1, index=0,
                        logprob=-1.2, top_alts=alts)
        assert ev.top_alts == alts
        assert len(ev.top_alts) == 2

    def test_disabled_means_no_alts(self):
        """K=0 produces ``top_alts is None``, not ``[]`` (per plan).
        ``None`` is the "alts not captured" signal subscribers
        ``?? null``-guard on; an empty list would be ambiguous between
        "captured but no alts" and "not captured."  The engine sticks
        to ``None`` on the K=0 path so the wire contract stays clean."""
        ev = TokenEvent(text="x", token_id=0, index=0, logprob=-1.0)
        assert ev.top_alts is None
        # And conversely: an explicitly empty list would be a distinct
        # shape the consumer could (in principle) treat differently.
        ev_empty = TokenEvent(text="x", token_id=0, index=0,
                              logprob=-1.0, top_alts=[])
        assert ev_empty.top_alts == []
        assert ev_empty.top_alts is not None


# ---------------------------------------------------------------------------
# GenerationResult.logprobs new inner shape
# ---------------------------------------------------------------------------


class TestGenerationResultLogprobs:
    def test_triple_shape_with_token_alts(self):
        """Inner shape post-phase-1: ``(token_id, logprob, list[TokenAlt])``
        — replaces the legacy ``(token_id, logprob, list[(id, lp)])``
        pair shape. OpenAI route reads ``alt.text`` directly without
        retokenizing.
        """
        alts = [TokenAlt(id=10, text=" a", logprob=-0.2),
                TokenAlt(id=11, text=" b", logprob=-1.1)]
        result = GenerationResult(
            text="hello", tokens=[1, 2], token_count=2,
            tok_per_sec=10.0, elapsed=0.2,
            logprobs=[(1, -0.4, alts), (2, -1.0, [])],
        )
        assert result.logprobs is not None
        first = result.logprobs[0]
        assert first[0] == 1
        assert first[1] == pytest.approx(-0.4)
        assert first[2][0].text == " a"


# ---------------------------------------------------------------------------
# LoomNode mean_logprob / mean_surprise — round-trip + finalize
# ---------------------------------------------------------------------------


class TestLoomNodeMeanLogprob:
    def test_default_none(self):
        node = LoomNode(id="n1", parent_id=None, role="user")
        assert node.mean_logprob is None
        assert node.mean_surprise is None

    def test_explicit_values(self):
        node = LoomNode(id="n1", parent_id=None, role="assistant",
                        mean_logprob=-1.5, mean_surprise=1.5)
        assert node.mean_logprob == pytest.approx(-1.5)
        assert node.mean_surprise == pytest.approx(1.5)

    def test_to_dict_includes_fields(self):
        node = LoomNode(id="n1", parent_id=None, role="assistant",
                        mean_logprob=-2.1, mean_surprise=2.1)
        d = node.to_dict()
        assert d["mean_logprob"] == pytest.approx(-2.1)
        assert d["mean_surprise"] == pytest.approx(2.1)

    def test_to_dict_carries_none_for_legacy(self):
        """Legacy nodes (created before the logit pass) have ``None`` —
        the wire payload still carries the key so downstream consumers
        get a deterministic shape regardless of capture status."""
        node = LoomNode(id="n1", parent_id=None, role="assistant")
        d = node.to_dict()
        assert "mean_logprob" in d and d["mean_logprob"] is None
        assert "mean_surprise" in d and d["mean_surprise"] is None

    def test_from_dict_roundtrip(self):
        node = LoomNode(id="n1", parent_id=None, role="assistant",
                        mean_logprob=-0.7, mean_surprise=0.7)
        restored = LoomNode.from_dict(node.to_dict())
        assert restored.mean_logprob == pytest.approx(-0.7)
        assert restored.mean_surprise == pytest.approx(0.7)

    def test_from_dict_missing_fields_default_none(self):
        """Replay from a pre-logit-pass transcript snapshot — the keys
        aren't present in the dict; ``from_dict`` defaults them to
        ``None`` so the legacy shape loads cleanly."""
        restored = LoomNode.from_dict({
            "id": "n1", "parent_id": None, "role": "assistant",
            "text": "hi",
        })
        assert restored.mean_logprob is None
        assert restored.mean_surprise is None


class TestFinalizeAssistantStampsLogprobs:
    def test_stamps_mean_logprob(self):
        """``LoomTree.finalize_assistant`` writes ``mean_logprob`` /
        ``mean_surprise`` onto the node when the caller supplies them
        (session computes from the engine's chosen-token logprob
        stream and passes through here)."""
        tree = LoomTree(model_id="test", session_id="s1")
        user_id = tree.add_user_turn("hi")
        node_id = tree.begin_assistant(user_id)
        tree.finalize_assistant(
            node_id,
            text="response",
            mean_logprob=-1.8,
            mean_surprise=1.8,
        )
        node = tree.nodes[node_id]
        assert node.mean_logprob == pytest.approx(-1.8)
        assert node.mean_surprise == pytest.approx(1.8)

    def test_none_when_not_supplied(self):
        """Backward compat: ``finalize_assistant`` without the new
        kwargs leaves the fields at their default ``None``. The session
        passes ``None`` when no on_token consumer / logprobs request
        was live for the gen, so legacy paths land cleanly."""
        tree = LoomTree(model_id="test", session_id="s1")
        user_id = tree.add_user_turn("hi")
        node_id = tree.begin_assistant(user_id)
        tree.finalize_assistant(node_id, text="response")
        node = tree.nodes[node_id]
        assert node.mean_logprob is None
        assert node.mean_surprise is None


# ---------------------------------------------------------------------------
# CLI / YAML wiring — ConfigFile.return_top_k
# ---------------------------------------------------------------------------


class TestConfigFileReturnTopK:
    def test_default_none(self):
        from saklas.cli.config_file import ConfigFile
        cfg = ConfigFile()
        assert cfg.return_top_k is None

    def test_load_valid(self, tmp_path):
        """YAML round-trip — ``return_top_k:`` lands on the dataclass."""
        from saklas.cli.config_file import ConfigFile
        p = tmp_path / "cfg.yaml"
        p.write_text("return_top_k: 8\n")
        cfg = ConfigFile.load(p)
        assert cfg.return_top_k == 8

    def test_load_zero(self, tmp_path):
        from saklas.cli.config_file import ConfigFile
        p = tmp_path / "cfg.yaml"
        p.write_text("return_top_k: 0\n")
        cfg = ConfigFile.load(p)
        assert cfg.return_top_k == 0

    def test_load_rejects_negative(self, tmp_path):
        from saklas.cli.config_file import ConfigFile, ConfigFileError
        p = tmp_path / "cfg.yaml"
        p.write_text("return_top_k: -1\n")
        with pytest.raises(ConfigFileError):
            ConfigFile.load(p)

    def test_load_rejects_above_256(self, tmp_path):
        from saklas.cli.config_file import ConfigFile, ConfigFileError
        p = tmp_path / "cfg.yaml"
        p.write_text("return_top_k: 300\n")
        with pytest.raises(ConfigFileError):
            ConfigFile.load(p)

    def test_load_rejects_non_int(self, tmp_path):
        from saklas.cli.config_file import ConfigFile, ConfigFileError
        p = tmp_path / "cfg.yaml"
        p.write_text('return_top_k: "eight"\n')
        with pytest.raises(ConfigFileError):
            ConfigFile.load(p)

    def test_load_rejects_bool(self, tmp_path):
        """YAML ``true`` would silently coerce to ``1`` without an
        explicit bool guard — reject it so users don't end up with
        K=1 from a typo'd boolean."""
        from saklas.cli.config_file import ConfigFile, ConfigFileError
        p = tmp_path / "cfg.yaml"
        p.write_text("return_top_k: true\n")
        with pytest.raises(ConfigFileError):
            ConfigFile.load(p)

    def test_compose_later_wins(self):
        from saklas.cli.config_file import ConfigFile, compose
        a = ConfigFile(return_top_k=4)
        b = ConfigFile(return_top_k=12)
        merged = compose([a, b])
        assert merged.return_top_k == 12

    def test_compose_none_passes_through(self):
        from saklas.cli.config_file import ConfigFile, compose
        a = ConfigFile(return_top_k=4)
        b = ConfigFile()  # None on every field
        merged = compose([a, b])
        assert merged.return_top_k == 4
