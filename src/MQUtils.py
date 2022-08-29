from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.core import UTCDateTime
from scipy import signal

from src.configure import *

from .utils import _get_datetime_ticks


def trim(
    mq,
    start,
    end,
    channel="ALL",
):
    """
    trimming

    Args:
        start (list[int]): start time ([year, month, day, hour, minute, second, microsecond])
        end (list[int]): end time ([year, month, day, hour, minute, second, microsecond])
        channel (str): {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
    """
    if channel == "ALL":
        channels = ["LPX", "LPY", "LPZ", "SPZ"]
    else:
        channels = [channel]

    for ch in channels:
        data = None
        if ch == "LPX":
            data, data_du = mq.lpx, mq.lpx_du
        elif ch == "LPY":
            data, data_du = mq.lpy, mq.lpy_du
        elif ch == "LPZ":
            data, data_du = mq.lpz, mq.lpz_du
        elif ch == "SPZ":
            data, data_du = mq.spz, mq.spz_du

        if data is None:
            continue

        starttime_before = data[0].stats.starttime
        starttime_after = UTCDateTime(*start)
        endtime_after = UTCDateTime(*end)
        datetimes = data[0].times()

        index = (datetimes >= starttime_after - starttime_before) & (
            datetimes <= endtime_after - starttime_before
        )
        data[0].data = data[0].data[index]
        data_du[0].data = data_du[0].data[index]

        data[0].stats.starttime = starttime_after


def du2phys(mq, channel="ALL", lower_th=None, upper_th=None):
    """
    convert DU into physical quantity

    Args:
        channel (str): {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
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

        if ch != "SPZ":
            data.detrend(type="linear")  # detrend
            data.detrend("demean")  # demean
            data.taper(0.05, type="cosine")
            data.filter("lowpass", freq=pre_high_freq, zerophase=True)
            data.filter("highpass", freq=pre_low_freq, zerophase=True)
            data.simulate(paz_remove=paz_AP)  # Remove response
            data.taper(0.05, type="cosine")
            data.differentiate(method="gradient")

            if (lower_th is None) or (upper_th is None):
                data.filter("lowpass", freq=post_high_freq, zerophase=True)
                data.filter("highpass", freq=post_low_freq, zerophase=True)

        elif ch == "SPZ":
            data.detrend(type="linear")  # detrend
            data.detrend("demean")  # demean
            data.taper(0.05, type="cosine")
            data.filter("lowpass", freq=pre_high_freq_sp, zerophase=True)
            data.filter("highpass", freq=pre_low_freq_sp, zerophase=True)
            data.simulate(paz_remove=paz_SP)  # Remove response
            data.taper(0.05, type="cosine")
            data.differentiate(method="gradient")

            if (lower_th is None) or (upper_th is None):
                data.filter("lowpass", freq=post_high_freq_sp, zerophase=True)
                data.filter("highpass", freq=post_low_freq_sp, zerophase=True)

        if (lower_th is not None) and (upper_th is not None):
            data.filter("lowpass", freq=upper_th, zerophase=True)
            data.filter("highpass", freq=lower_th, zerophase=True)


def remove_noise(
    mq, channel="ALL", method="envelope", times=5, verbose=0, is_widget=False
):
    """
    remove noise

    Args:
        channel (str): {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
        method (str): {'envelope', 'ewm'}
        times (int): times of approve the method
        verbose (int): visualized if >0
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

        d = np.array(data[0].data)
        n_plots = 5 + times

        if verbose >= 1:
            fig = plt.figure(figsize=(8 if is_widget else 24, 3 * n_plots))
            title = "Waveform (DU)"
            _plotting(fig, d, [n_plots, 1, 1], title, color="black")

        # remove extreme noise
        LOWER_LIMIT, UPPER_LIMIT = 10, 1014
        d[d < LOWER_LIMIT] = 0
        d[d > UPPER_LIMIT] = 0

        # detrend with window
        window = 1000 if ch == "SPZ" else 100
        d[d != 0] = _detrend_rolling(d[d != 0], window=window, step=window)
        if verbose >= 1:
            title = "Remove skipping noise + Detrend"
            _plotting(fig, d, [n_plots, 1, 2], title, color="blue")

        if method == "envelope":
            window = 3000
            env_mul, std_mul = 3, 1.5
            for t in range(times):
                if verbose >= 1:
                    title = f"Remove noise with envelope ({t+1} times)"
                    _plotting(fig, d, [n_plots, 1, 3 + t], title, color="red")
                threshold = (
                    np.convolve(
                        np.ones(window) / window, abs(signal.hilbert(d)), mode="same"
                    )
                    * env_mul
                    + np.std(d) * std_mul
                )
                d[np.abs(d) > threshold] = 0

                if verbose >= 1:
                    ax = _plotting(fig, d, [n_plots, 1, 3 + t], title, color="blue")
                    ax.plot(threshold, color="#32CD32")
                    ax.plot(-threshold, color="#32CD32")
                    ax.set_yticks([])

        elif method == "ewm":
            window = 1000 if ch == "SPZ" else 100
            std_mul = 4
            d = pd.Series(d)
            for t in range(times):
                if verbose >= 1:
                    title = f"Remove noise with ewm ({t+1} times)"
                    _plotting(fig, d, [n_plots, 1, 3 + t], title, color="red")
                threshold = (d.ewm(span=window).std() * std_mul)[window:].reset_index(
                    drop=True
                )
                d[:-window][d.abs()[:-window] > threshold] = 0

                if verbose >= 1:
                    ax = _plotting(fig, d, [n_plots, 1, 3 + t], title, color="blue")
                    ax.plot(threshold, color="#32CD32")
                    ax.plot(-threshold, color="#32CD32")

            d = np.array(d)
            d = d[:-window]  # remove tail data

        else:
            assert False

        if verbose >= 1:
            title = "Result"
            _plotting(fig, d, [n_plots, 1, 3 + t + 1], title, color="black")

            ticks, datetime_ticks = _get_datetime_ticks(data)
            plt.xticks(ticks, datetime_ticks)
            plt.show()

        data[0].data = d


def culc_sta_lta(
    mq,
    channel="SPZ",
    fc=1.0,
    n=None,
    m=None,
    l=None,
    verbose=0,
    is_widget=False,
    title=None,
):
    """
    calculate sta/lta

    Args:
        mq (MQData): moonquake data
        channel (str): {'LPX', 'LPY', 'LPZ', 'SPZ'}
        fc (float): 1.0 or center frequency
        n (float): parameter of tl
        m (float): parameter of ts
        l (float): parameter of lag
        verbose (int): set >0 if visualize
        is_widget (bool): use widget if True
        title (str): figure title
    """
    data = None
    if channel == "LPX":
        data = mq.lpx
    elif channel == "LPY":
        data = mq.lpy
    elif channel == "LPZ":
        data = mq.lpz
    elif channel == "SPZ":
        data = mq.spz

    if data is None:
        return

    sampling_rate = data[0].stats.sampling_rate
    d = pd.Series(data[0].data).abs()

    tl = round(sampling_rate * (1 / fc) * n) if n else round(sampling_rate * 150)
    ts = round(sampling_rate * (1 / fc) * m) if m else round(sampling_rate * 50)
    lag = round(sampling_rate * (1 / fc) * l) if l else round(sampling_rate * 56)

    sta, lta = d.rolling(ts).mean(), d.rolling(tl).mean()
    sta, lta = sta[~pd.isnull(sta)], lta[~pd.isnull(lta)]

    sta = sta[tl + lag :]
    lta = lta[: sta.shape[0]]

    sta, lta = np.array(sta), np.array(lta)
    result = sta / lta
    result = result * signal.cosine(len(result))

    if verbose > 0:
        fig, ax1 = plt.subplots(figsize=(8 if is_widget else 24, 4))

        color = "red"
        ax1.set_ylabel("STA/LTA", color=color)
        ax1.plot(result, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(-5, 100)

        color = "black"
        ax2 = ax1.twinx()
        ax2.set_ylabel("m/s", color=color)
        ax2.plot(d[tl + ts + lag :], color=color, alpha=0.3)

        fig.tight_layout()
        plt.xticks([])
        if title:
            plt.title(title)
        plt.show()

    return result


def culc_peak_run_length(
    slta,
    threshold,
    run_length_th=None,
    verbose=0,
    signal=None,
    is_widget=False,
    title=None,
):
    """
    calculate run length

    Args:
        slta (ndarray): sta/lta
        threshold (float): threshold of sta/lta
        run_length_th (float): threshold of run length
        verbose (int): set >0 if visualize
        signal (ndarray): plot signal
        is_widget (bool): use widget if True
        title (str): figure title
    """
    comp_sec, lengths = _culc_run_length(slta >= threshold)
    run_length = lengths[comp_sec]

    start_args = np.array(
        [
            np.where(slta >= threshold)[0][np.sum(run_length[:i])]
            for i in range(len(run_length))
        ]
    )
    end_args = np.array(
        [
            np.where(slta >= threshold)[0][np.sum(run_length[: i + 1]) - 1]
            for i in range(len(run_length))
        ]
    )

    if run_length_th is not None:
        target_args = run_length >= run_length_th
        run_length = run_length[target_args]
        start_args = start_args[target_args]
        end_args = end_args[target_args]

    if verbose > 0:
        print(run_length)

        x_figsize = 8 if is_widget else 24
        fig, ax1 = plt.subplots(figsize=(x_figsize, 8))
        slta_th = deepcopy(slta)
        slta_th[slta < threshold] = None

        ax1.set_ylabel("STA/LTA", color="blue")
        ax1.plot(slta, color="blue")
        ax1.plot(slta_th, color="red")
        ax1.plot(np.repeat(threshold, len(slta)), color="#32CD32")
        ax1.tick_params(axis="y", labelcolor="blue")

        if signal is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel("nm/s", color="black")
            ax2.plot(pd.Series(signal[-len(slta) :]).abs(), color="black", alpha=0.2)

        fig.tight_layout()
        if title:
            plt.title(title)
        plt.show()

    return run_length, start_args, end_args


def _detrend_rolling(data, window, step=1):
    n_data = len(data)
    i = 0
    while True:
        data[i : i + window] = signal.detrend(data[i : i + window])
        i += step
        if i >= n_data:
            break
    return data


def _plotting(fig, data, add_subplot_args, title, color="blue"):
    ax = fig.add_subplot(*add_subplot_args)
    ax.plot(data, color=color)
    ax.set_xticks([])
    plt.title(title)
    return ax


def _culc_run_length(sequence):
    diff_seq = np.diff(sequence)

    newdata = np.append(True, diff_seq != 0)
    comp_seq = sequence[newdata]

    comp_seq_index = np.where(newdata)[0]
    comp_seq_index = np.append(comp_seq_index, len(sequence))
    lengths = np.diff(comp_seq_index)

    return comp_seq, lengths
