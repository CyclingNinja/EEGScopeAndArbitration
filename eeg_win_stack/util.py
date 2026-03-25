"""Compatibility layer for older imports during migration."""

from eeg_win_stack.io.eeg_loading import custom_crop, load_brainvision_as_windows
from eeg_win_stack.io.labeling import relabel
from eeg_win_stack.tools.filters import (
    check_inf,
    exclude_by_name,
    remove_same,
    remove_tuab_from_dataset,
    select_by_channel,
    select_by_duration,
    select_labeled,
)
from eeg_win_stack.tools.metrics import (
    MCC,
    con_mat,
    find_all_zero,
    timecost,
    top1,
    top1_prob,
    top1_prob1,
    weight_function,
)
from eeg_win_stack.tools.paths import (
    findall,
    get_full_filelist,
    natural_key,
    read_all_file_names,
    session_key,
    time_key,
)
from eeg_win_stack.tools.splits import split_data