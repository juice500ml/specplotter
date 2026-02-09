"""Tests for SpecPlotter class."""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

from specplotter import SpecPlotter


class TestSpecPlotter:
    """Test cases for SpecPlotter class."""

    def test_init_default(self):
        """Test SpecPlotter initialization with default parameters."""
        plotter = SpecPlotter()
        assert plotter.mode == "wideband"
        assert plotter.assumed_rate == 16000
        assert plotter.fnotch == 60
        assert plotter.notchQ == 30
        assert plotter.coeff == 0.97
        assert plotter.window_size == 0.004
        assert plotter.window_stride == 0.001

    def test_init_narrowband(self):
        """Test SpecPlotter initialization with narrowband mode."""
        plotter = SpecPlotter(mode="narrowband")
        assert plotter.mode == "narrowband"
        assert plotter.window_size == 0.025
        assert plotter.window_stride == 0.01

    def test_init_custom_parameters(self):
        """Test SpecPlotter initialization with custom parameters."""
        plotter = SpecPlotter(
            mode="wideband", sample_rate=22050, fnotch=50, notchQ=25, db_spread=80
        )
        assert plotter.assumed_rate == 22050
        assert plotter.fnotch == 50
        assert plotter.notchQ == 25
        assert plotter.db_spread == 80

    def test_init_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'wideband' or 'narrowband'"):
            SpecPlotter(mode="invalid")

    def test_compute_spectrogram(self):
        """Test compute_spectrogram returns expected dictionary."""
        plotter = SpecPlotter()
        # Generate a simple test signal (1 second of sine wave at 440 Hz)
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        features = plotter.compute_spectrogram(signal)

        assert isinstance(features, dict)
        assert "processed_signal" in features
        assert "spectrogram" in features
        assert "zcr" in features
        assert "total_energy" in features
        assert "lowfreq_energy" in features

        # Check that all features are numpy arrays
        assert isinstance(features["processed_signal"], np.ndarray)
        assert isinstance(features["spectrogram"], np.ndarray)
        assert isinstance(features["zcr"], np.ndarray)
        assert isinstance(features["total_energy"], np.ndarray)
        assert isinstance(features["lowfreq_energy"], np.ndarray)

        # Check shapes are reasonable
        assert features["processed_signal"].shape[0] == signal.shape[0]
        assert len(features["spectrogram"].shape) == 2
        assert len(features["zcr"].shape) == 1
        assert len(features["total_energy"].shape) == 1
        assert len(features["lowfreq_energy"].shape) == 1

    def test_plot_spectrogram_only(self):
        """Test plot_spectrogram creates figure without errors."""
        plotter = SpecPlotter()
        # Generate a simple test signal
        duration = 0.1  # Short signal for faster tests
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        fig, ax = plotter.plot_spectrogram(signal)

        assert fig is not None
        assert ax is not None

    def test_plot_with_all_components(self):
        """Test plot with all components enabled."""
        plotter = SpecPlotter()
        # Generate a simple test signal
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        fig, axes = plotter.plot(
            signal,
            show_zcr=True,
            show_total_energy=True,
            show_lowfreq_energy=True,
            show_waveform=True,
        )

        assert fig is not None
        assert isinstance(axes, dict)
        assert "spectrogram" in axes
        assert "zcr" in axes
        assert "total_energy" in axes
        assert "lowfreq_energy" in axes
        assert "waveform" in axes

    def test_plot_with_custom_axis(self):
        """Test plot with custom matplotlib axis."""
        import matplotlib.pyplot as plt

        plotter = SpecPlotter()
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        fig, ax = plt.subplots()
        fig_result, ax_result = plotter.plot_spectrogram(signal, ax=ax)

        assert fig_result == fig
        assert ax_result == ax
        plt.close(fig)

    def test_plot_save_to_file(self, tmp_path):
        """Test plot can save to file."""
        plotter = SpecPlotter()
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        output_file = tmp_path / "test_output.png"
        plotter.plot_spectrogram(signal, outfile=str(output_file))

        assert output_file.exists()

    def test_individual_plot_methods(self):
        """Test individual plotting methods."""
        import matplotlib.pyplot as plt

        plotter = SpecPlotter()
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        features = plotter.compute_spectrogram(signal)

        # Test each plotting method
        fig, ax = plt.subplots()
        plotter.plot_zcr(features["zcr"], ax=ax)
        plt.close(fig)

        fig, ax = plt.subplots()
        plotter.plot_total_energy(features["total_energy"], ax=ax)
        plt.close(fig)

        fig, ax = plt.subplots()
        plotter.plot_lowfreq_energy(features["lowfreq_energy"], ax=ax)
        plt.close(fig)

        fig, ax = plt.subplots()
        plotter.plot_waveform(features["processed_signal"], ax=ax)
        plt.close(fig)

    def test_plot_with_list_of_axes(self):
        """Test plot with list of axes."""
        import matplotlib.pyplot as plt

        plotter = SpecPlotter()
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        fig, axes = plt.subplots(3, 1)
        axes_list = list(axes)

        fig_result, axes_dict = plotter.plot(
            signal, ax=axes_list, show_zcr=True, show_waveform=True
        )

        assert fig_result == fig
        assert len(axes_dict) == 3
        plt.close(fig)

    def test_plot_axes_count_mismatch(self):
        """Test that mismatched axis count raises ValueError."""
        import matplotlib.pyplot as plt

        plotter = SpecPlotter()
        duration = 0.1
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * 440 * t)

        fig, axes = plt.subplots(2, 1)  # Only 2 axes
        axes_list = list(axes)

        # Should raise error because we need 3 axes (zcr, spectrogram, waveform)
        with pytest.raises(ValueError, match="Number of axes"):
            plotter.plot(signal, ax=axes_list, show_zcr=True, show_waveform=True)
        plt.close(fig)
