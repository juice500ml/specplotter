# SpecPlotter

A Python package for creating wideband and narrowband spectrograms with comprehensive signal analysis features.

## Features

- **Wideband and narrowband spectrogram generation** - Switch between analysis modes
- **Modular signal processing** - Compute individual features independently
- **Flexible plotting** - Choose which components to display
- **Zero crossing rate analysis** - Optional ZCR visualization
- **Energy analysis** - Total energy and low-frequency energy calculations
- **Waveform visualization** - Optional waveform display
- **Fully configurable** - All parameters are customizable

## Installation

```bash
pip install specplotter
```

## Quick Start

### Command Line Interface

The easiest way to use SpecPlotter is via the command line:

```bash
# Basic spectrogram (PNG)
specplotter audio.wav -o output.png

# Basic spectrogram (PDF)
specplotter audio.wav -o output.pdf

# Full analysis with all components
specplotter audio.wav -o output.pdf --zcr --total-energy --lowfreq-energy --waveform

# Narrowband mode
specplotter audio.wav -o output.png --mode narrowband
```

### Python API

```python
from specplotter import SpecPlotter
import librosa

# Load an audio file
signal, sr = librosa.load('audio.wav', sr=16000)

# Create a SpecPlotter instance (default: wideband mode)
plotter = SpecPlotter()

# Plot spectrogram only (default behavior)
plotter.plot_spectrogram(signal)

# Or save to file
plotter.plot_spectrogram(signal, outfile='spectrogram.png')
```

## Usage Examples

### Basic Usage

```python
from specplotter import SpecPlotter
import librosa

signal, sr = librosa.load('audio.wav', sr=16000)
plotter = SpecPlotter()

# Default: spectrogram only
plotter.plot_spectrogram(signal)
```

### Wideband vs Narrowband

```python
# Wideband mode (default)
plotter_wide = SpecPlotter(mode='wideband')

# Narrowband mode
plotter_narrow = SpecPlotter(mode='narrowband')

# Plot with different modes
plotter_wide.plot_spectrogram(signal)
plotter_narrow.plot_spectrogram(signal)
```

### Flexible Plotting with Optional Components

```python
# Plot with all components
plotter.plot(
    signal,
    show_zcr=True,
    show_total_energy=True,
    show_lowfreq_energy=True,
    show_waveform=True
)

# Just spectrogram and waveform
plotter.plot(signal, show_waveform=True)

# Spectrogram with zero crossing rate
plotter.plot(signal, show_zcr=True)
```

### Using Custom Axes

```python
import matplotlib.pyplot as plt

# Single axis (spectrogram only)
fig, ax = plt.subplots()
plotter.plot_spectrogram(signal, ax=ax)

# List of axes matching number of plots
fig, axes = plt.subplots(3, 1)  # For zcr, spectrogram, waveform
plotter.plot(
    signal,
    ax=list(axes),
    show_zcr=True,
    show_waveform=True
)

# Full plot with all components
fig, axes = plt.subplots(5, 1)
plotter.plot(
    signal,
    ax=list(axes),
    show_zcr=True,
    show_total_energy=True,
    show_lowfreq_energy=True,
    show_waveform=True
)
```

**Note:** When providing a list of axes, they must be in this order:
1. `zcr` (if `show_zcr=True`)
2. `total_energy` (if `show_total_energy=True`)
3. `lowfreq_energy` (if `show_lowfreq_energy=True`)
4. `spectrogram` (always included)
5. `waveform` (if `show_waveform=True`)

### Computing Features Without Plotting

```python
# Compute all features
features = plotter.compute_spectrogram(signal)

# Access individual features
processed_signal = features['processed_signal']
spectrogram = features['spectrogram']
zcr = features['zcr']
total_energy = features['total_energy']
lowfreq_energy = features['lowfreq_energy']

# Use features independently
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plotter._plot_spectrogram_on_axis(spectrogram, len(signal), ax=ax)
```

### Individual Plotting Methods

```python
features = plotter.compute_spectrogram(signal)

# Plot individual components
fig, ax = plt.subplots()
plotter.plot_zcr(features['zcr'], ax=ax)

fig, ax = plt.subplots()
plotter.plot_total_energy(features['total_energy'], ax=ax)

fig, ax = plt.subplots()
plotter.plot_lowfreq_energy(features['lowfreq_energy'], ax=ax)

fig, ax = plt.subplots()
plotter.plot_waveform(features['processed_signal'], ax=ax)
```

## Configuration

### Mode Selection

The `mode` parameter sets default window sizes for wideband or narrowband analysis:

- **Wideband** (default): `window_size=0.004`, `window_stride=0.001`
- **Narrowband**: `window_size=0.025`, `window_stride=0.01`

### Customizing Parameters

All parameters can be customized when creating a SpecPlotter instance:

```python
plotter = SpecPlotter(
    mode='wideband',              # 'wideband' or 'narrowband'
    sample_rate=16000,            # Sample rate (Hz)
    fnotch=60,                    # Notch filter frequency (Hz)
    notchQ=30,                    # Notch filter Q factor
    preemphasis_coeff=0.97,       # Pre-emphasis coefficient
    window_size=0.004,            # Window size (seconds), optional
    window_stride=0.001,          # Window stride (seconds), optional
    n_fft=1024,                   # Number of FFT points
    window=scipy.signal.windows.hamming,  # Window function
    db_spread=60,                 # Dynamic range (dB)
    db_cutoff=3,                  # Minimum dB value
    fig_height=10,                # Figure height (inches)
    inches_per_sec=10,            # Horizontal scaling
    zcr_smoothing_std=6,          # ZCR smoothing std dev
    zcr_smoothing_size=41,       # ZCR smoothing kernel size
    lowfreq_min=125,              # Low freq energy min (Hz)
    lowfreq_max=750,              # Low freq energy max (Hz)
)
```

### Overriding Mode Defaults

You can override mode defaults by explicitly providing `window_size` and `window_stride`:

```python
# Use wideband mode but with custom window settings
plotter = SpecPlotter(
    mode='wideband',
    window_size=0.005,   # Override default
    window_stride=0.002  # Override default
)
```

## Command Line Interface

SpecPlotter includes a command-line interface for quick spectrogram generation.

### Basic Usage

```bash
specplotter <input_file> -o <output_file>
```

The output format (PNG or PDF) is automatically determined from the file extension.

### Options

**Required:**
- `input_file`: Path to input WAV file
- `-o, --output`: Output file path (must have .png or .pdf extension)

**Analysis Mode:**
- `--mode {wideband,narrowband}`: Analysis mode (default: wideband)

**Additional Plots:**
- `--all`: Show all additional plots (zcr, total-energy, lowfreq-energy, waveform)
- `--zcr`: Show zero crossing rate plot
- `--total-energy`: Show total energy plot
- `--lowfreq-energy`: Show low frequency energy plot
- `--waveform`: Show waveform plot

**Audio Settings:**
- `--sample-rate FLOAT`: Sample rate in Hz (default: 16000)

**Processing Parameters:**
- `--fnotch FLOAT`: Notch filter frequency in Hz (default: 60)
- `--notch-q FLOAT`: Notch filter Q factor (default: 30)
- `--db-spread FLOAT`: Dynamic range in dB (default: 60)
- `--db-cutoff FLOAT`: Minimum dB value to display (default: 3)

### Examples

```bash
# Basic spectrogram
specplotter audio.wav -o spectrogram.png

# Full analysis (all components)
specplotter audio.wav -o analysis.pdf --all

# Narrowband mode with custom settings
specplotter audio.wav -o output.pdf --mode narrowband --db-spread 80

# Custom sample rate
specplotter audio.wav -o output.png --sample-rate 22050

# European line noise (50 Hz instead of 60 Hz)
specplotter audio.wav -o output.pdf --fnotch 50
```

### Help

For full help and all options:

```bash
specplotter --help
```

## API Reference

### `SpecPlotter.__init__()`

Initialize SpecPlotter with configurable parameters. See Configuration section above for all parameters.

### `compute_spectrogram(signal)`

Compute spectrogram and related features.

**Parameters:**
- `signal` (np.ndarray): Input audio signal

**Returns:**
- `dict`: Dictionary containing:
  - `'processed_signal'`: Preprocessed signal
  - `'spectrogram'`: Clipped log spectrogram
  - `'zcr'`: Zero crossing rate
  - `'total_energy'`: Total energy envelope
  - `'lowfreq_energy'`: Low frequency energy envelope

### `plot(signal, ax=None, show_zcr=False, show_total_energy=False, show_lowfreq_energy=False, show_waveform=False, outfile=None, **kwargs)`

Plot spectrogram with optional additional features.

**Parameters:**
- `signal` (np.ndarray): Input audio signal
- `ax` (Axes or list of Axes, optional): Matplotlib axes to plot on
- `show_zcr` (bool): Whether to show zero crossing rate plot
- `show_total_energy` (bool): Whether to show total energy plot
- `show_lowfreq_energy` (bool): Whether to show low frequency energy plot
- `show_waveform` (bool): Whether to show waveform plot
- `outfile` (str, optional): If provided, save figure to file
- `**kwargs`: Additional keyword arguments passed to plotting functions

**Returns:**
- `tuple`: (figure, axes_dict)

### `plot_spectrogram(signal, ax=None, outfile=None, **kwargs)`

Plot spectrogram only (convenience method).

**Parameters:**
- `signal` (np.ndarray): Input audio signal
- `ax` (Axes, optional): Matplotlib axes to plot on
- `outfile` (str, optional): If provided, save figure to file
- `**kwargs`: Additional keyword arguments

**Returns:**
- `tuple`: (figure, axes)

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- scipy >= 1.5.0
- librosa >= 0.8.0
- matplotlib >= 3.3.0

## License

MIT License

## Versioning

This package uses [python-semantic-release](https://python-semantic-release.readthedocs.io/) for automatic version management based on [Conventional Commits](https://www.conventionalcommits.org/).

### How it works

The version is automatically determined from your commit messages:
- `fix:` or `fix(scope):` → patch version bump (0.0.1 → 0.0.2)
- `feat:` or `feat(scope):` → minor version bump (0.0.1 → 0.1.0)
- `feat!:` or `fix!:` or `BREAKING CHANGE:` → major version bump (0.0.1 → 1.0.0)

### Creating a new release

Simply push commits with conventional commit messages to the main branch:

```bash
git commit -m "feat: add new spectrogram visualization feature"
git push origin main
```

The GitHub Actions workflow will:
1. Analyze your commits since the last release
2. Determine the appropriate version bump
3. Update version numbers in `pyproject.toml` and `__init__.py`
4. Create a git tag
5. Generate/update CHANGELOG.md
6. Create a GitHub release
7. Build and publish to PyPI (if `PYPI_API_TOKEN` is configured)

### Commit message format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat: add support for custom color maps"
git commit -m "fix: correct frequency calculation in spectrogram"
git commit -m "feat!: change API for plot_spectrogram method"
git commit -m "docs: update installation instructions"
```

## Testing

Run tests using pytest:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_specplotter.py
```

The test suite includes:
- Unit tests for SpecPlotter initialization and configuration
- Tests for spectrogram computation
- Tests for plotting functionality
- Tests for CLI interface
- Tests for error handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Setting up pre-commit hooks

This project uses pre-commit hooks to ensure code quality. To set them up:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Or run manually
pre-commit run --all-files
```

The pre-commit hooks will automatically:
- Remove trailing whitespace
- Fix end-of-file issues
- Check YAML, JSON, and TOML syntax
- Format code with ruff
- Run ruff linting (with auto-fix)
- Check for merge conflicts

### Before submitting

1. Run the test suite: `pytest`
2. Run pre-commit hooks: `pre-commit run --all-files`
3. Ensure all tests pass
4. Ensure code formatting is correct
