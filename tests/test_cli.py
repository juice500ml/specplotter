"""Tests for CLI interface."""

import subprocess
import sys


def test_cli_help():
    """Test that CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "specplotter.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Generate spectrograms from audio files" in result.stdout
    assert "--output" in result.stdout
    assert "--mode" in result.stdout


def test_cli_missing_args():
    """Test that CLI fails with missing required arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "specplotter.cli"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()
