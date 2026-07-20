"""Enable ``python -m eeg_win_stack`` — delegates to the CLI's ``main``."""

import sys

from eeg_win_stack.cli import main

if __name__ == "__main__":
    sys.exit(main())
