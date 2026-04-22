import time
import warnings
from itertools import product

import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datautil import load_concat_dataset
from braindecode.datasets import BaseConcatDataset, TUH, TUHAbnormal
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    exponential_moving_standardize,
    preprocess,
    scale,
)

from eeg_win_stack.evaluation.detail_exporter import TrainingDetailExporter
from eeg_win_stack.evaluation.evaluator import EEGEvaluator
from eeg_win_stack.io.eeg_loading import custom_crop
from eeg_win_stack.io.labeling import relabel
from eeg_win_stack.tools.filters import (
    remove_tuab_from_dataset,
    select_by_channel,
    select_by_duration,
    select_labeled,
)
from eeg_win_stack.tools.splits import split_data
from eeg_win_stack.training.trainer import EEGTrainer, build_model
from batch_test_hyperparameters import *
from train_and_eval_config import *

warnings.filterwarnings("once")
pd.set_option("display.max_columns", 10)

LOG_COLUMNS = [
    "train_loss", "valid_loss", "train_accuracy", "valid_accuracy",
    "etl_time", "model_training_time",
    "test_acc", "test_precision", "test_recall",
    "n_repetition", "random_state", "tuab", "tueg", "n_tuab", "n_tueg",
    "n_load", "preload", "window_len_s",
    "tuab_path", "tueg_path", "saved_data", "saved_path",
    "saved_windows_data", "saved_windows_path",
    "load_saved_data", "load_saved_windows", "bandpass_filter",
    "low_cut_hz", "high_cut_hz", "standardization", "factor_new",
    "init_block_size", "n_jobs", "n_classes", "lr", "weight_decay",
    "batch_size", "n_epochs", "tmin", "tmax", "multiple", "sec_to_cut",
    "duration_recording_sec", "max_abs_val", "sampling_freq", "test_on_eval",
    "split_way", "train_size", "valid_size", "test_size", "shuffle",
    "model_name", "final_conv_length", "window_stride_samples",
    "relabel_dataset", "relabel_label", "channels", "dropout",
    "precision_per_recording", "recall_per_recording", "acc_per_recording",
    "mcc", "mcc_per_recording", "activation", "remove_attribute",
]

EEGEvaluator.write_header(log_path, LOG_COLUMNS)

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

for (
    random_state, tuab, tueg, n_tuab, n_tueg, n_load, preload, window_len_s,
    tuab_path, tueg_path, saved_data, saved_path, saved_windows_data, saved_windows_path,
    load_saved_data, load_saved_windows, bandpass_filter, low_cut_hz, high_cut_hz,
    standardization, factor_new, init_block_size, n_jobs, tmin, tmax, multiple, sec_to_cut,
    duration_recording_sec, max_abs_val, sampling_freq, test_on_eval, split_way,
    train_size, valid_size, test_size, shuffle, window_stride_samples,
    relabel_dataset, relabel_label, channels, remove_attribute, activation,
) in product(
    RANDOM_STATE, TUAB, TUEG, N_TUAB, N_TUEG, N_LOAD, PRELOAD,
    WINDOW_LEN_S, TUAB_PATH, TUEG_PATH, SAVED_DATA, SAVED_PATH, SAVED_WINDOWS_DATA,
    SAVED_WINDOWS_PATH, LOAD_SAVED_DATA, LOAD_SAVED_WINDOWS, BANDPASS_FILTER,
    LOW_CUT_HZ, HIGH_CUT_HZ, STANDARDIZATION, FACTOR_NEW, INIT_BLOCK_SIZE, N_JOBS,
    TMIN, TMAX, MULTIPLE, SEC_TO_CUT,
    DURATION_RECORDING_SEC, MAX_ABS_VAL, SAMPLING_FREQ, TEST_ON_VAL, SPLIT_WAY,
    TRAIN_SIZE, VALID_SIZE, TEST_SIZE, SHUFFLE, WINDOW_STRIDE_SAMPLES,
    RELABEL_DATASET, RELABEL_LABEL, CHANNELS, REMOVE_ATTRIBUTE, ACTIVATION,
):
    mne.set_log_level(mne_log_level)
    torch.set_num_threads(n_jobs)

    data_loading_start = time.time()
    window_len_samples = window_len_s * sampling_freq

    if load_saved_windows:
        load_ids = list(range(n_load)) if n_load else None
        windows_ds = load_concat_dataset(
            path=saved_windows_path,
            preload=False,
            ids_to_load=load_ids,
            target_name="pathological",
            n_jobs=1,
        )
    else:
        if load_saved_data:
            load_ids = list(range(n_load)) if n_load else None
            ds = load_concat_dataset(
                path=saved_path,
                preload=preload,
                ids_to_load=load_ids,
                target_name="pathological",
            )
        else:
            tuab_ids = list(range(n_tuab)) if n_tuab else None
            ds_tuab = TUHAbnormal(tuab_path, recording_ids=tuab_ids, target_name="pathological", preload=preload)

            if tueg:
                tueg_ids = list(range(n_tueg)) if n_tueg else None
                ds_tueg = TUH(tueg_path, recording_ids=tueg_ids, target_name="pathological", preload=preload)
                if tuab:
                    ds_tueg = remove_tuab_from_dataset(ds_tueg, tuab_path)

            ds = BaseConcatDataset(
                ([i for i in ds_tuab.datasets] if tuab else [])
                + ([j for j in ds_tueg.datasets] if tueg else [])
            )
            ds = select_by_duration(ds, tmin, tmax)

            for i in range(len(relabel_label)):
                ds.set_description(relabel(ds, relabel_label[i], relabel_dataset[i]), overwrite=True)

            ds = select_labeled(ds)
            ds = select_by_channel(ds, channels)

            preprocessors = [
                Preprocessor("pick_types", eeg=True, meg=False, stim=False),
                Preprocessor("pick_channels", ch_names=channels, ordered=True),
                Preprocessor(fn="resample", sfreq=sampling_freq),
                Preprocessor(
                    custom_crop,
                    tmin=sec_to_cut,
                    tmax=duration_recording_sec + sec_to_cut,
                    include_tmax=False,
                    apply_on_array=False,
                ),
                Preprocessor(scale, factor=1e6, apply_on_array=True),
                Preprocessor(np.clip, a_min=-max_abs_val, a_max=max_abs_val, apply_on_array=True),
            ]
            if multiple:
                preprocessors.append(Preprocessor(scale, factor=multiple, apply_on_array=True))
            if bandpass_filter:
                preprocessors.append(Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz))
            if standardization:
                preprocessors.append(
                    Preprocessor(
                        exponential_moving_standardize,
                        factor_new=factor_new,
                        init_block_size=init_block_size,
                    )
                )

            preprocess(ds, preprocessors, save_dir=saved_path, overwrite=False, n_jobs=n_jobs)

        fs = ds.datasets[0].raw.info["sfreq"]
        window_len_samples = int(fs * window_len_s)
        if not window_stride_samples:
            window_stride_samples = window_len_samples

        windows_ds = create_fixed_length_windows(
            ds,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=window_len_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=True,
            preload=preload,
            drop_bad_windows=True,
        )
        for ds in windows_ds.datasets:
            ds.windows.drop_bad()
            assert ds.windows.preload == preload

        if saved_windows_data:
            windows_ds.save(saved_windows_path, True)

    train_set, valid_set, test_set = split_data(
        windows_ds, split_way, train_size, valid_size, test_size, shuffle, random_state, remove_attribute
    )
    etl_time = time.time() - data_loading_start
    n_channels = windows_ds[0][0].shape[0]

    for (repetition_idx, n_classes, lr, weight_decay, batch_size, n_epochs, model_name, final_conv_length, dropout) in product(
        range(N_REPETITIONS), N_CLASSES, LR, WEIGHT_DECAY, BATCH_SIZE, N_EPOCHS, MODEL_NAME, FINAL_CONV_LENGTH, DROPOUT
    ):
        if shuffle and repetition_idx > 0:
            train_set, valid_set, test_set = split_data(
                windows_ds, split_way, train_size, valid_size, test_size, shuffle, random_state + repetition_idx
            )

        mne.set_log_level(mne_log_level)

        model = build_model(
            model_name,
            n_channels,
            n_classes,
            window_len_samples,
            sampling_freq,
            activation=activation,
            dropout=dropout,
            final_conv_length=final_conv_length,
            deep4_n_filters_time=deep4_n_filters_time,
            deep4_n_filters_spat=deep4_n_filters_spat,
            deep4_filter_time_length=deep4_filter_time_length,
            deep4_pool_time_length=deep4_pool_time_length,
            deep4_pool_time_stride=deep4_pool_time_stride,
            deep4_n_filters_2=deep4_n_filters_2,
            deep4_filter_length_2=deep4_filter_length_2,
            deep4_n_filters_3=deep4_n_filters_3,
            deep4_filter_length_3=deep4_filter_length_3,
            deep4_n_filters_4=deep4_n_filters_4,
            deep4_filter_length_4=deep4_filter_length_4,
            deep4_first_pool_mode=deep4_first_pool_mode,
            deep4_later_pool_mode=deep4_later_pool_mode,
            shallow_n_filters_time=shallow_n_filters_time,
            shallow_filter_time_length=shallow_filter_time_length,
            shallow_n_filters_spat=shallow_n_filters_spat,
            shallow_pool_time_length=shallow_pool_time_length,
            shallow_pool_time_stride=shallow_pool_time_stride,
            shallow_split_first_layer=shallow_split_first_layer,
            shallow_batch_norm=shallow_batch_norm,
            shallow_batch_norm_alpha=shallow_batch_norm_alpha,
            tcn_n_blocks=tcn_n_blocks,
            tcn_n_filters=tcn_n_filters,
            tcn_kernel_size=tcn_kernel_size,
            tcn_add_log_softmax=tcn_add_log_softmax,
            tcn_last_layer_type=tcn_last_layer_type,
            vit_patch_size=vit_patch_size,
            vit_dim=vit_dim,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            vit_mlp_dim=vit_mlp_dim,
            vit_emb_dropout=vit_emb_dropout,
        )
        if cuda:
            model.cuda()

        trainer = EEGTrainer(
            model,
            train_set,
            valid_set,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
            test_on_eval=test_on_eval,
            earlystopping=earlystopping,
        )

        if not test_model:
            clf, history_df = trainer.fit()
            trainer.save(
                f"./saved_models/{model_name}{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}params.pt"
            )
        else:
            clf = trainer.load(f"./saved_models/{params[repetition_idx]}")
            history_df = None

        evaluator = EEGEvaluator(clf, test_set)
        metrics = evaluator.evaluate()

        if not test_model:
            last = history_df.iloc[-1]
            result_prefix = [last["train_loss"], last["valid_loss"], last["train_accuracy"], last["valid_accuracy"]]
        else:
            result_prefix = ["test_model", "test_model", "test_model", "test_model"]

        row_values = [
            *result_prefix,
            etl_time,
            trainer.training_time,
            metrics["acc"],
            metrics["precision"],
            metrics["recall"],
            repetition_idx,
            random_state,
            tuab,
            tueg,
            n_tuab,
            n_tueg,
            n_load,
            preload,
            window_len_s,
            tuab_path,
            tueg_path,
            saved_data,
            saved_path,
            saved_windows_data,
            saved_windows_path,
            load_saved_data,
            load_saved_windows,
            bandpass_filter,
            low_cut_hz,
            high_cut_hz,
            standardization,
            factor_new,
            init_block_size,
            n_jobs,
            n_classes,
            lr,
            weight_decay,
            batch_size,
            n_epochs,
            tmin,
            tmax,
            multiple,
            sec_to_cut,
            duration_recording_sec,
            max_abs_val,
            sampling_freq,
            test_on_eval,
            split_way,
            train_size,
            valid_size,
            test_size,
            shuffle,
            model_name,
            final_conv_length,
            window_stride_samples,
            relabel_dataset,
            relabel_label,
            channels,
            dropout,
            metrics["precision_per_recording"],
            metrics["recall_per_recording"],
            metrics["acc_per_recording"],
            metrics["mcc"],
            metrics["mcc_per_recording"],
            activation,
            remove_attribute,
        ]

        evaluator.write_results(log_path, row_values, history_df=history_df)

        if train_whole_dataset_again:
            TrainingDetailExporter(clf, train_set, valid_set, test_set).export("./training_detail.csv")
