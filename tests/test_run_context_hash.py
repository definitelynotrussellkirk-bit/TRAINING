# tests/test_run_context_hash.py
"""Tests for RunContext hash consistency."""

from core.run_context import get_run_context


def test_context_hash_matches_identity_summary():
    """
    identity_summary() should embed the same context_hash
    that context_hash() returns.
    """
    ctx = get_run_context()
    summary = ctx.identity_summary()

    assert "context_hash" in summary
    assert summary["context_hash"] == ctx.context_hash()


def test_context_hash_is_stable():
    """
    Multiple calls to context_hash() should return the same value
    for the same RunContext.
    """
    ctx = get_run_context()
    hash1 = ctx.context_hash()
    hash2 = ctx.context_hash()

    assert hash1 == hash2


def test_context_hash_is_16_chars():
    """context_hash should be a 16-character hex string."""
    ctx = get_run_context()
    h = ctx.context_hash()

    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
