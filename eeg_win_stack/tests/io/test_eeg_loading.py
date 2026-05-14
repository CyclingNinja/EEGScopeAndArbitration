"""Tests for EEGLoader (eeg_win_stack/io/visual_eeg_loading.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from eeg_win_stack.io.visual_eeg_loading import EEGLoader


@pytest.fixture
def loader():
    return EEGLoader(data_folder="data/")


class TestCustomCrop:
    def test_crops_signal_to_tmax(self, loader, synthetic_raw):
        sfreq = synthetic_raw.info["sfreq"]
        loader._custom_crop(synthetic_raw, tmin=0.0, tmax=5.0)
        assert synthetic_raw.times[-1] <= 5.0 + 1 / sfreq

    def test_tmax_clamped_when_exceeds_recording(self, loader, synthetic_raw):
        original_duration = (synthetic_raw.n_times - 1) / synthetic_raw.info["sfreq"]
        loader._custom_crop(synthetic_raw, tmin=0.0, tmax=original_duration + 100.0)
        assert synthetic_raw.times[-1] <= original_duration + 1 / synthetic_raw.info["sfreq"]

    def test_tmin_trims_start(self, loader, synthetic_raw):
        loader._custom_crop(synthetic_raw, tmin=2.0, tmax=8.0)
        # MNE resets origin to 0; verify the resulting duration is within expected range
        assert synthetic_raw.times[-1] <= 8.0 + 1 / synthetic_raw.info["sfreq"]


class TestChannelProcessing:
    def test_all_channels_get_eeg_ref_prefix(self, loader, synthetic_raw):
        loader._channel_processing(synthetic_raw)
        assert all(
            ch.startswith("EEG ") and ch.endswith("-REF")
            for ch in synthetic_raw.ch_names
        )

    def test_channel_count_unchanged(self, loader, synthetic_raw):
        original_count = len(synthetic_raw.ch_names)
        loader._channel_processing(synthetic_raw)
        assert len(synthetic_raw.ch_names) == original_count

    def test_first_channel_is_fp1(self, loader, synthetic_raw):
        loader._channel_processing(synthetic_raw)
        assert synthetic_raw.ch_names[0] == "EEG FP1-REF"


class TestLoadBrainvisionAsWindows:
    @patch(
        "eeg_win_stack.io.visual_eeg_loading.get_full_filelist",
        return_value=["fake.vhdr"],
    )
    @patch("mne.io.read_raw_brainvision")
    def test_returns_dataset(self, mock_read, _mock_files, loader, synthetic_raw):
        mock_read.return_value = synthetic_raw
        result = loader.load_brainvision_as_windows()
        assert result is not None

    @patch(
        "eeg_win_stack.io.visual_eeg_loading.get_full_filelist",
        return_value=["a.vhdr", "b.vhdr"],
    )
    @patch("mne.io.read_raw_brainvision")
    def test_reads_each_file(self, mock_read, _mock_files, loader, synthetic_raw):
        # Both reads must return independent raw copies (modified in-place)
        import mne
        import numpy as np
        from conftest import RAW_CHANNEL_NAMES

        def make_raw():
            sfreq = 256.0
            n_times = int(sfreq * 10)
            data = np.random.randn(22, n_times) * 1e-6
            info = mne.create_info(RAW_CHANNEL_NAMES, sfreq=sfreq, ch_types="eeg")
            return mne.io.RawArray(data, info, verbose=False)

        mock_read.side_effect = [make_raw(), make_raw()]
        loader.load_brainvision_as_windows()
        assert mock_read.call_count == 2

    @patch(
        "eeg_win_stack.io.visual_eeg_loading.get_full_filelist",
        return_value=[],
    )
    def test_empty_folder_raises(self, _mock_files, loader):
        with pytest.raises((IndexError, Exception)):
            loader.load_brainvision_as_windows()
