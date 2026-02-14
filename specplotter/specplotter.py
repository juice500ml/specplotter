import numpy as np
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple, Dict, Literal, Callable, Union, List


class SpecPlotter(object):
    def __init__(
        self,
        mode: Literal["wideband", "narrowband"] = "wideband",
        sample_rate: float = 16000,
        fnotch: float = 60,
        notchQ: float = 30,
        preemphasis_coeff: float = 0.97,
        window_size: Optional[float] = None,
        window_stride: Optional[float] = None,
        n_fft: int = 1024,
        window: Callable = scipy.signal.windows.hamming,
        db_spread: float = 60,
        db_cutoff: float = 3,
        fig_height: float = 10,
        inches_per_sec: float = 10,
        zcr_smoothing_std: float = 6,
        zcr_smoothing_size: int = 41,
        lowfreq_min: float = 125,
        lowfreq_max: float = 750,
    ):
        """
        Initialize SpecPlotter with configurable parameters.

        Parameters
        ----------
        mode : {'wideband', 'narrowband'}, default 'wideband'
            Analysis mode. Sets default window_size and window_stride:
            - 'wideband': window_size=0.004, window_stride=0.001
            - 'narrowband': window_size=0.025, window_stride=0.01
        sample_rate : float, default 16000
            Assumed sample rate of input signals (Hz)
        fnotch : float, default 60
            Notch filter frequency (Hz) for removing line noise
        notchQ : float, default 30
            Notch filter Q factor (quality factor)
        preemphasis_coeff : float, default 0.97
            Pre-emphasis filter coefficient (0-1)
        window_size : float, optional
            Window size in seconds. If None, uses mode defaults:
            - wideband: 0.004
            - narrowband: 0.025
        window_stride : float, optional
            Window stride/hop size in seconds. If None, uses mode defaults:
            - wideband: 0.001
            - narrowband: 0.01
        n_fft : int, default 1024
            Number of FFT points
        window : callable, default scipy.signal.windows.hamming
            Window function to use for STFT
        db_spread : float, default 60
            Dynamic range in dB for spectrogram display
        db_cutoff : float, default 3
            Minimum dB value to display (clips below this)
        fig_height : float, default 10
            Figure height in inches
        inches_per_sec : float, default 10
            Horizontal scaling (inches per second of audio)
        zcr_smoothing_std : float, default 6
            Standard deviation for Gaussian smoothing of zero crossing rate
        zcr_smoothing_size : int, default 41
            Size of Gaussian kernel for zero crossing rate smoothing
        lowfreq_min : float, default 125
            Minimum frequency (Hz) for low frequency energy calculation
        lowfreq_max : float, default 750
            Maximum frequency (Hz) for low frequency energy calculation
        """
        # Set mode-dependent defaults
        if mode == "wideband":
            default_window_size = 0.004
            default_window_stride = 0.001
        elif mode == "narrowband":
            default_window_size = 0.025
            default_window_stride = 0.01
        else:
            raise ValueError(f"mode must be 'wideband' or 'narrowband', got '{mode}'")

        # Store parameters
        self.mode = mode
        self.assumed_rate = sample_rate
        self.fnotch = fnotch
        self.notchQ = notchQ
        self.coeff = preemphasis_coeff
        self.window_size = (
            window_size if window_size is not None else default_window_size
        )
        self.window_stride = (
            window_stride if window_stride is not None else default_window_stride
        )
        self.n_fft = n_fft
        self.window = window
        self.db_spread = db_spread
        self.db_cutoff = db_cutoff
        self.fig_height = fig_height
        self.inches_per_sec = inches_per_sec
        self.zcr_smoothing_std = zcr_smoothing_std
        self.zcr_smoothing_size = zcr_smoothing_size
        self.lowfreq_min = lowfreq_min
        self.lowfreq_max = lowfreq_max

        # Compute derived parameters
        self.hop_length = int(self.assumed_rate * self.window_stride)
        self.win_length = int(self.assumed_rate * self.window_size)

    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess signal: remove mean, apply notch filter, and pre-emphasis."""
        y = signal - signal.mean()
        b, a = scipy.signal.iirnotch(self.fnotch, self.notchQ, self.assumed_rate)
        y = scipy.signal.lfilter(b, a, y)
        y = np.append(y[0], y[1:] - self.coeff * y[:-1])
        return y

    def _compute_zcr(self, signal: np.ndarray) -> np.ndarray:
        """Compute zero crossing rate with smoothing."""
        g = scipy.signal.windows.gaussian(
            self.zcr_smoothing_size, std=self.zcr_smoothing_std
        )
        g = g / g.sum()
        zcr = librosa.feature.zero_crossing_rate(
            signal, frame_length=self.win_length, hop_length=self.hop_length
        )
        zcr = np.convolve(zcr[0], g, mode="same")
        zcr = zcr - zcr.min()
        return zcr

    def _compute_stft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute STFT and power spectrogram."""
        stft = librosa.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        power_spec = np.abs(stft) ** 2
        return stft, power_spec

    def _compute_total_energy(self, power_spec: np.ndarray) -> np.ndarray:
        """Compute total energy with smoothing and normalization."""
        g = scipy.signal.windows.gaussian(
            self.zcr_smoothing_size, std=self.zcr_smoothing_std
        )
        g = g / g.sum()
        total_energy = 10 * np.log10(np.sum(power_spec, axis=0))
        total_energy = total_energy - total_energy.max()
        total_energy = np.clip(total_energy, -1 * self.db_spread, 0)
        total_energy = total_energy - total_energy.min()
        total_energy = np.convolve(total_energy, g, mode="same")
        total_energy = total_energy - total_energy.max()
        return total_energy

    def _compute_lowfreq_energy(self, power_spec: np.ndarray) -> np.ndarray:
        """Compute low frequency energy with smoothing and normalization."""
        g = scipy.signal.windows.gaussian(
            self.zcr_smoothing_size, std=self.zcr_smoothing_std
        )
        g = g / g.sum()
        f0 = int(
            np.round((self.lowfreq_min / self.assumed_rate * 0.5) * power_spec.shape[0])
        )
        f1 = int(
            np.round((self.lowfreq_max / self.assumed_rate * 0.5) * power_spec.shape[0])
        )
        lowfreq_energy = 10 * np.log10(np.sum(power_spec[f0:f1, :], axis=0))
        lowfreq_energy = lowfreq_energy - lowfreq_energy.max()
        lowfreq_energy = np.clip(lowfreq_energy, -1 * self.db_spread, 0)
        lowfreq_energy = lowfreq_energy - lowfreq_energy.min()
        lowfreq_energy = np.convolve(lowfreq_energy, g, mode="same")
        lowfreq_energy = lowfreq_energy - lowfreq_energy.max()
        return lowfreq_energy

    def _compute_spectrogram(self, power_spec: np.ndarray) -> np.ndarray:
        """Compute log spectrogram with clipping."""
        logspec = librosa.power_to_db(power_spec, ref=np.max)
        logspec = np.flipud(logspec)
        clipped_logspec = np.clip(logspec, -1 * self.db_spread, -1 * self.db_cutoff)
        return clipped_logspec

    def compute_spectrogram(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute spectrogram and related features.

        Parameters
        ----------
        signal : np.ndarray
            Input audio signal

        Returns
        -------
        dict
            Dictionary containing:
            - 'processed_signal': Preprocessed signal
            - 'spectrogram': Clipped log spectrogram
            - 'zcr': Zero crossing rate
            - 'total_energy': Total energy envelope
            - 'lowfreq_energy': Low frequency energy envelope
        """
        # Preprocess signal
        y = self._preprocess_signal(signal)

        # Compute ZCR
        zcr = self._compute_zcr(signal - signal.mean())

        # Compute STFT
        stft, power_spec = self._compute_stft(y)

        # Compute energy features
        total_energy = self._compute_total_energy(power_spec)
        lowfreq_energy = self._compute_lowfreq_energy(power_spec)

        # Compute spectrogram
        spectrogram = self._compute_spectrogram(power_spec)

        return {
            "processed_signal": y,
            "spectrogram": spectrogram,
            "zcr": zcr,
            "total_energy": total_energy,
            "lowfreq_energy": lowfreq_energy,
        }

    def plot_zcr(self, zcr: np.ndarray, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """Plot zero crossing rate."""
        if ax is None:
            ax = plt.gca()

        color = kwargs.get("color", "gray")
        ax.fill_between(np.arange(len(zcr)), zcr, y2=zcr.min(), color=color)
        ax.margins(0, 0)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("kHz")
        ax.annotate("Zero Crossing Rate", (10, 0.6))
        return ax

    def plot_total_energy(
        self, total_energy: np.ndarray, ax: Optional[Axes] = None, **kwargs
    ) -> Axes:
        """Plot total energy."""
        if ax is None:
            ax = plt.gca()

        color = kwargs.get("color", "gray")
        ax.fill_between(
            np.arange(len(total_energy)),
            total_energy,
            y2=total_energy.min(),
            color=color,
        )
        ax.margins(0, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("dB")
        ax.annotate("Total Energy", (10, (-15 / 40) * np.abs(total_energy.min())))
        return ax

    def plot_lowfreq_energy(
        self, lowfreq_energy: np.ndarray, ax: Optional[Axes] = None, **kwargs
    ) -> Axes:
        """Plot low frequency energy."""
        if ax is None:
            ax = plt.gca()

        color = kwargs.get("color", "gray")
        ax.fill_between(
            np.arange(len(lowfreq_energy)),
            lowfreq_energy,
            y2=lowfreq_energy.min(),
            color=color,
        )
        ax.margins(0, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("dB")
        ax.annotate(
            f"Energy: {self.lowfreq_min} to {self.lowfreq_max} Hz",
            (10, -(15 / 40) * np.abs(lowfreq_energy.min())),
        )
        return ax

    def _plot_spectrogram_on_axis(
        self,
        spectrogram: np.ndarray,
        signal_length: int,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        """Plot spectrogram on given axis (internal helper method)."""
        if ax is None:
            ax = plt.gca()

        extent = [
            0,
            signal_length / self.assumed_rate,
            0,
            self.assumed_rate / 2000,
        ]  # Convert x to seconds, y to kHz
        n_sec = signal_length / self.assumed_rate

        cmap = kwargs.get("cmap", "gist_gray_r")
        ax.imshow(spectrogram, cmap=cmap, extent=extent, aspect="auto")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_xticks(np.arange(0, n_sec, 0.1))
        ax.tick_params(labelbottom=False, labelleft=True, labelright=True)
        ax.grid(color="k", linestyle="dotted", linewidth=0.5)
        show_annotation = kwargs.get("show_annotation", True)
        if show_annotation:
            mode_label = "Wide Band" if self.mode == "wideband" else "Narrow Band"
            ax.annotate(f"{mode_label} Spectrogram", (0.01, 7.7))
        return ax

    def plot_waveform(
        self, signal: np.ndarray, ax: Optional[Axes] = None, **kwargs
    ) -> Axes:
        """Plot waveform."""
        if ax is None:
            ax = plt.gca()

        n_sec = signal.shape[0] / self.assumed_rate
        linewidth = kwargs.get("linewidth", 0.25)
        color = kwargs.get("color", "k")

        ax.plot(signal, linewidth=linewidth, color=color)
        ax.margins(0, 0)
        ticks = np.arange(0, n_sec, 0.1)
        ticklabs = ["%.1f" % z for z in ticks]
        ax.set_xticks(ticks=self.assumed_rate * ticks, labels=ticklabs)
        ax.set_yticks([])
        ax.set_xlabel("Time (seconds)")
        ax.annotate("Waveform", (200, 0.3 * signal.max()))
        return ax

    def plot(
        self,
        signal: np.ndarray,
        ax: Optional[Union[Axes, List[Axes]]] = None,
        show_zcr: bool = False,
        show_total_energy: bool = False,
        show_lowfreq_energy: bool = False,
        show_waveform: bool = False,
        show_annotation: bool = True,
        outfile: Optional[str] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Dict[str, Axes]]:
        """
        Plot spectrogram with optional additional features.

        Parameters
        ----------
        signal : np.ndarray
            Input audio signal
        ax : plt.Axes or list of plt.Axes, optional
            Matplotlib axes to plot on. Can be:
            - None: creates a new figure
            - Single Axes: plots only spectrogram on that axis
            - List of Axes: must match the number of plots (in order: zcr,
              total_energy, lowfreq_energy, spectrogram, waveform)
        show_zcr : bool, default False
            Whether to show zero crossing rate plot
        show_total_energy : bool, default False
            Whether to show total energy plot
        show_lowfreq_energy : bool, default False
            Whether to show low frequency energy plot
        show_waveform : bool, default False
            Whether to show waveform plot
        show_annotation : bool, default True
            Whether to show the mode annotation on the spectrogram
        outfile : str, optional
            If provided, save figure to file instead of displaying
        **kwargs
            Additional keyword arguments passed to plotting functions

        Returns
        -------
        figure : plt.Figure
            The matplotlib figure
        axes : dict
            Dictionary of axes used for plotting

        Raises
        ------
        ValueError
            If list of axes is provided but length doesn't match number of plots
        """
        # Compute features
        features = self.compute_spectrogram(signal)

        # Determine number of subplots needed and their order
        plot_order = []
        if show_zcr:
            plot_order.append("zcr")
        if show_total_energy:
            plot_order.append("total_energy")
        if show_lowfreq_energy:
            plot_order.append("lowfreq_energy")
        plot_order.append("spectrogram")  # Always included
        if show_waveform:
            plot_order.append("waveform")

        n_plots = len(plot_order)

        # Handle axes input
        if ax is None:
            n_sec = signal.shape[0] / self.assumed_rate
            if n_plots == 1:
                # Only spectrogram
                fig = plt.figure(figsize=(n_sec * self.inches_per_sec, self.fig_height))
                gs = fig.add_gridspec(nrows=1, ncols=1)
                axes = {"spectrogram": fig.add_subplot(gs[0, 0])}
            else:
                # Multiple plots with appropriate ratios
                height_ratios = []
                for plot_name in plot_order:
                    if plot_name == "spectrogram":
                        height_ratios.append(12)
                    else:
                        height_ratios.append(1)

                fig = plt.figure(figsize=(n_sec * self.inches_per_sec, self.fig_height))
                gs = fig.add_gridspec(
                    nrows=n_plots, ncols=1, height_ratios=height_ratios, hspace=0.05
                )

                axes = {}
                for idx, plot_name in enumerate(plot_order):
                    axes[plot_name] = fig.add_subplot(gs[idx, 0])
        elif isinstance(ax, list):
            # List of axes provided
            if len(ax) != n_plots:
                raise ValueError(
                    f"Number of axes ({len(ax)}) must match number of plots ({n_plots}). "
                    f"Expected order: {', '.join(plot_order)}"
                )
            # Map axes to plot names in order
            axes = dict(zip(plot_order, ax))
            fig = ax[0].figure
        else:
            # Single axis provided - only plot spectrogram
            if n_plots > 1:
                raise ValueError(
                    f"Single axis provided but {n_plots} plots requested. "
                    f"Provide a list of {n_plots} axes or set additional plot flags to False."
                )
            fig = ax.figure
            axes = {"spectrogram": ax}

        # Plot components in order
        for plot_name in plot_order:
            if plot_name == "zcr":
                self.plot_zcr(features["zcr"], axes["zcr"], **kwargs)
            elif plot_name == "total_energy":
                self.plot_total_energy(
                    features["total_energy"], axes["total_energy"], **kwargs
                )
            elif plot_name == "lowfreq_energy":
                self.plot_lowfreq_energy(
                    features["lowfreq_energy"], axes["lowfreq_energy"], **kwargs
                )
            elif plot_name == "spectrogram":
                self._plot_spectrogram_on_axis(
                    features["spectrogram"],
                    signal.shape[0],
                    axes["spectrogram"],
                    show_annotation=show_annotation,
                    **kwargs,
                )
            elif plot_name == "waveform":
                self.plot_waveform(
                    features["processed_signal"], axes["waveform"], **kwargs
                )

        # Add x-axis label to spectrogram if no waveform
        if "waveform" not in axes:
            axes["spectrogram"].set_xlabel("Time (seconds)")
            axes["spectrogram"].tick_params(labelbottom=True)

        if outfile:
            fig.savefig(outfile, bbox_inches="tight")
        elif ax is None:
            plt.show()

        return fig, axes

    def plot_spectrogram(
        self,
        signal: np.ndarray,
        ax: Optional[Axes] = None,
        outfile: Optional[str] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Axes]:
        """
        Plot spectrogram only (default behavior).

        This is a convenience method that calls plot() with default settings.

        Parameters
        ----------
        signal : np.ndarray
            Input audio signal
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, creates a new figure.
        outfile : str, optional
            If provided, save figure to file instead of displaying
        **kwargs
            Additional keyword arguments passed to plotting functions

        Returns
        -------
        figure : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The axes used for plotting
        """
        fig, axes = self.plot(
            signal,
            ax=ax,
            show_zcr=False,
            show_total_energy=False,
            show_lowfreq_energy=False,
            show_waveform=False,
            outfile=outfile,
            **kwargs,
        )
        return fig, axes["spectrogram"]
