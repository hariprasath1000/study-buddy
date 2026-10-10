"""StudyBuddy entry point.

Run with:  uv run streamlit run main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # lets Community Cloud find the package

from studybuddy.ui import main  # noqa: E402

if __name__ == "__main__":
    main()