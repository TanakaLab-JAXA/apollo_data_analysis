import os
from copy import deepcopy

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy import signal

from src.configure import *

from .utils import _get_datetime_ticks


def plot_du(mq, channel="ALL", save=False, path=""):
    """
    plot DU data

    Args:
        mq (MQData): Moonquake Data
        channel (str): {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
        save (bool): save figures if True
        path (str): output path of figures if save
    """
    if channel == "ALL":
        channels = ["LPX", "LPY", "LPZ", "SPZ"]
    else:
        channels = [channel]

    for ch in channels:
        data_du = None
        if ch == "LPX":
            data_du = mq.lpx_du
        elif ch == "LPY":
            data_du = mq.lpy_du
        elif ch == "LPZ":
            data_du = mq.lpz_du
        elif ch == "SPZ":
            data_du = mq.spz_du

        if data_du is None:
            continue

        if save:
            os.makedirs(path, exist_ok=True)
            data_du.plot(outfile=f"{path}/{ch}.png")
        else:
            data_du.plot()
            plt.show()


def plot_comparison_with_du(
    mq,
    channel="ALL",
    is_du_detrend=False,
    is_phys=False,
    save=False,
    path="",
    is_widget=False,
):
    """
    compare current data with DU data

    Args:
        mq (MQData): Moonquake Data
        channel (str): {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
        is_phys (bool): The current signal is physical quantity or not
        save (bool): save figures if True
        path (str): output path of figures if save
        is_widget (bool): use widget if True
    """
    if channel == "ALL":
        channels = ["LPX", "LPY", "LPZ", "SPZ"]
    else:
        channels = [channel]

    for ch in channels:
        data, data_du = None, None
        if ch == "LPX":
            data = mq.lpx
            data_du = deepcopy(mq.lpx_du)
        elif ch == "LPY":
            data = mq.lpy
            data_du = deepcopy(mq.lpy_du)
        elif ch == "LPZ":
            data = mq.lpz
            data_du = deepcopy(mq.lpz_du)
        elif ch == "SPZ":
            data = mq.spz
            data_du = deepcopy(mq.spz_du)

        if data is None:
            continue

        # init
        fig = plt.figure(figsize=(8 if is_widget else 24, 8))
        ticks, datetime_ticks = _get_datetime_ticks(data)

        # # plot DU data
        ax = fig.add_subplot(2, 1, 1)
        if is_du_detrend:
            data_du.detrend(type="linear")
        ax.plot(data_du[0].data, color="black")
        ax.set_title("Waveform (DU)")
        plt.xticks(ticks, datetime_ticks)
        ax.set_xlabel("Datetime (UTC)")
        ax.set_ylabel("$DU$")
        ax.grid()

        # # plot current data
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(data[0].data, color="black")
        ax.set_title("Waveform (Preprocessed)")
        plt.xticks(ticks, datetime_ticks)
        ax.set_xlabel("Datetime (UTC)")
        ax.set_ylabel("$m/s$" if is_phys else "$DU$")
        ax.grid()

        plt.subplots_adjust(hspace=0.4)

        if save:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/{ch}.png")
        else:
            plt.show()


def plot_spectrogram(mq, channel="ALL", save=False, path="", is_widget=False):
    """
    plot spectrogram

    Args:
        mq (MQData): Moonquake Data
        channel (str): {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
        save (bool): save figures if True
        path (str): output path of figures if save
        is_widget (bool): use widget if True
    """
    if channel == "ALL":
        channels = ["LPX", "LPY", "LPZ", "SPZ"]
    else:
        channels = [channel]

    for ch in channels:
        data = None
        if ch == "LPX":
            data = mq.lpx
        elif ch == "LPY":
            data = mq.lpy
        elif ch == "LPZ":
            data = mq.lpz
        elif ch == "SPZ":
            data = mq.spz

        if data is None:
            continue

        # init figure
        fig = plt.figure(figsize=(8 if is_widget else 24, 12))
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.axis("off")

        ticks, datetime_ticks = _get_datetime_ticks(data)

        # plot waveform
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(data[0].data)
        ax.set_xticks(ticks, [None for _ in ticks])
        ax.set_xmargin(0)
        ax.set_title("Waveform")
        ax.set_ylabel("$m/s$")
        ax.grid(which="major")

        # plot spectrogram
        ax = fig.add_subplot(2, 1, 2)

        # low freq
        if ch != "SPZ":
            f, t, sxx = signal.spectrogram(
                data[0].data * 1e9,
                nfft=int(f_lp * times),
                nperseg=int(f_lp * times),
                fs=int(f_lp),
                noverlap=int(f_lp * times / 2),
                scaling="density",
                mode="psd",
                window=("hamming"),
            )
            plt.pcolormesh(
                t,
                f,
                np.sqrt(sxx),
                cmap=cmap,
                norm=colors.LogNorm(vmin=psd_min, vmax=psd_max / 2),
            )
            ax.set_ylim(0.1, 3)
            ax.set_yticks([0.1, 0.5, 1.0, 2.0])
            plt.title("Spectrogram (Low Freq)")

        # high freq
        if ch == "SPZ":
            f, t, sxx = signal.spectrogram(
                data[0].data * 1e9,
                nfft=int(f_sp * times),
                nperseg=int(f_sp * times),
                fs=int(f_sp),
                noverlap=int(f_sp * times / 2),
                scaling="density",
                mode="psd",
                window=("hamming"),
            )
            plt.pcolormesh(
                t,
                f,
                np.sqrt(sxx),
                cmap=cmap,
                norm=colors.LogNorm(vmin=psd_min, vmax=psd_max / 2),
            )
            ax.set_ylim(1, 26)
            ax.set_yticks([0.1, 0.5, 1.0, 5.0, 10, 20])
            plt.title("Spectrogram (High Freq)")

        # other settings
        SEP = 8
        n_t = len(t)
        t_ticks = [i * n_t // SEP for i in range(SEP)] + [n_t - 1]
        ax.set_xticks(t[t_ticks], datetime_ticks)

        ax.set_xlabel("Datetime (UTC)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0.5, 16)
        ax.set_yscale("log")
        ax.set_yticks([1, 4, 8])
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.get_yaxis().set_tick_params(which="minor", size=0)

        plt.colorbar(orientation="horizontal", label="PSD ($nm/s/√Hz$)", aspect=80)
        plt.grid(which="major")
        plt.subplots_adjust(hspace=0.1)

        if save:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/{ch}.png")
        else:
            plt.show()
