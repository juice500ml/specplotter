#!/usr/bin/env python3
"""Command-line interface for SpecPlotter."""

import argparse
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for CLI

from .specplotter import SpecPlotter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate spectrograms from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic spectrogram (PNG)
  specplotter audio.wav -o output.png

  # Basic spectrogram (PDF)
  specplotter audio.wav -o output.pdf

  # Full analysis with all components
  specplotter audio.wav -o output.pdf --all

  # Narrowband mode
  specplotter audio.wav -o output.png --mode narrowband

  # Custom sample rate
  specplotter audio.wav -o output.png --sample-rate 22050
        """,
    )

    parser.add_argument("input_file", type=str, help="Path to input WAV file")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file path (PNG or PDF based on extension)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["wideband", "narrowband"],
        default="wideband",
        help="Analysis mode: wideband (default) or narrowband",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all additional plots (zcr, total-energy, lowfreq-energy, waveform)",
    )

    parser.add_argument(
        "--zcr", action="store_true", help="Show zero crossing rate plot"
    )

    parser.add_argument(
        "--total-energy", action="store_true", help="Show total energy plot"
    )

    parser.add_argument(
        "--lowfreq-energy", action="store_true", help="Show low frequency energy plot"
    )

    parser.add_argument("--waveform", action="store_true", help="Show waveform plot")

    parser.add_argument(
        "--fnotch",
        type=float,
        default=60,
        help="Notch filter frequency in Hz (default: 60)",
    )

    parser.add_argument(
        "--notch-q", type=float, default=30, help="Notch filter Q factor (default: 30)"
    )

    parser.add_argument(
        "--db-spread", type=float, default=60, help="Dynamic range in dB (default: 60)"
    )

    parser.add_argument(
        "--db-cutoff",
        type=float,
        default=3,
        help="Minimum dB value to display (default: 3)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() not in [".wav", ".wave"]:
        print(
            f"Warning: Input file may not be a WAV file: {args.input_file}",
            file=sys.stderr,
        )

    # Validate output file extension
    output_path = Path(args.output)
    output_ext = output_path.suffix.lower()
    if output_ext not in [".png", ".pdf"]:
        print(
            f"Error: Output file must have .png or .pdf extension, got: {output_ext}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load audio file
    try:
        print(f"Loading audio file: {args.input_file}")
        signal, sr = librosa.load(str(input_path), sr=int(args.sample_rate))
        print(f"Loaded with sample rate: {sr} Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}", file=sys.stderr)
        sys.exit(1)

    # Create SpecPlotter instance
    try:
        plotter = SpecPlotter(
            mode=args.mode,
            sample_rate=sr,
            fnotch=args.fnotch,
            notchQ=args.notch_q,
            db_spread=args.db_spread,
            db_cutoff=args.db_cutoff,
        )
    except Exception as e:
        print(f"Error creating SpecPlotter: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate plot
    try:
        # Handle --all flag
        show_zcr = args.zcr or args.all
        show_total_energy = args.total_energy or args.all
        show_lowfreq_energy = args.lowfreq_energy or args.all
        show_waveform = args.waveform or args.all

        print("Generating spectrogram...")
        print(f"  Mode: {args.mode}")
        print("  Components: spectrogram", end="")
        if show_zcr:
            print(" + zcr", end="")
        if show_total_energy:
            print(" + total_energy", end="")
        if show_lowfreq_energy:
            print(" + lowfreq_energy", end="")
        if show_waveform:
            print(" + waveform", end="")
        print()

        plotter.plot(
            signal,
            show_zcr=show_zcr,
            show_total_energy=show_total_energy,
            show_lowfreq_energy=show_lowfreq_energy,
            show_waveform=show_waveform,
            outfile=str(output_path),
        )

        print(f"Successfully saved to: {args.output}")

    except Exception as e:
        print(f"Error generating plot: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
