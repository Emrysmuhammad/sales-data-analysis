"""Check that the Python environment is ready.

Run:  python -m dataviz.check_setup
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    print(f"Python      {sys.version.split()[0]}")

    try:
        import matplotlib
        import numpy as np
        import pandas as pd
    except ImportError as error:
        print(f"Missing package: {error.name}")
        print("Install everything with:  pip install -r requirements-dev.txt")
        return 1

    print(f"numpy       {np.__version__}")
    print(f"pandas      {pd.__version__}")
    print(f"matplotlib  {matplotlib.__version__}")

    # Draw into a file instead of a window, so this also works on any machine
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dataviz import config

    config.apply_plot_style()
    frame = pd.DataFrame({"x": np.arange(6), "y": np.arange(6) ** 2})
    axes = frame.plot(x="x", y="y", title="Setup check", marker="o")

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "setup_check.png"
        axes.figure.savefig(target)
        saved = target.exists() and target.stat().st_size > 0
    plt.close("all")

    if not saved:
        print("A test chart could not be saved. Check your matplotlib installation.")
        return 1

    print("\nAll good: pandas and matplotlib are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())