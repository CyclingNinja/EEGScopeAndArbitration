"""EEG loading and preprocessing helpers."""

from __future__ import annotations

import mne
import numpy as np
from braindecode.datasets import create_from_X_y

from eeg_win_stack.tools.paths import get_full_filelist
from eeg_win_stack.tools.logging import get_logger

logger = get_logger(__name__)


def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    tmax = min((raw.n_times - 1) / raw.info["sfreq"], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


def load_brainvision_as_windows(data_folder):
    paths = get_full_filelist(data_folder, ".vhdr")
    logger.debug("BrainVision paths: %s", paths)

    def channel_processing(raw):
        for c in raw.ch_names:
            raw.rename_channels({c: "EEG " + c.upper() + "-REF"})
        raw.rename_channels({"EEG T7-REF": "EEG T3-REF"})
        raw.rename_channels({"EEG T8-REF": "EEG T4-REF"})
        raw.rename_channels({"EEG P7-REF": "EEG T5-REF"})
        raw.rename_channels({"EEG P8-REF": "EEG T6-REF"})
        raw.reorder_channels([
            "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF",
            "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF",
            "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF", "EEG T5-REF",
            "EEG T6-REF", "EEG FZ-REF", "EEG PZ-REF", "EEG FC1-REF", "EEG FC2-REF", "EEG CP1-REF", "EEG CP2-REF"
        ])
        return raw

    parts = [channel_processing(mne.io.read_raw_brainvision(path, preload=True, verbose=False)) for path in paths]

    def generate_cz(raw_data):
        return np.row_stack((raw_data[:18], (raw_data[18] + raw_data[19] + raw_data[20] + raw_data[21]) / 4))

    X = [generate_cz(raw.get_data()) for raw in parts]
    y = [1 for raw in parts]
    sfreq = parts[0].info["sfreq"]

    channels = [
        "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF",
        "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF",
        "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF", "EEG T5-REF",
        "EEG T6-REF", "EEG FZ-REF", "EEG PZ-REF", "EEG CZ-REF",
    ]

    windows_dataset = create_from_X_y(
        X,
        y,
        drop_last_window=False,
        sfreq=sfreq,
        ch_names=channels,
        window_stride_samples=6000,
        window_size_samples=6000,
    )
    return windows_dataset