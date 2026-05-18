"""Load and inspect the raw dataset."""

import logging

import pandas as pd
from pathlib import Path
from .config import DATA_PATH

logger = logging.getLogger(__name__)


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the flight fare CSV. Raises FileNotFoundError with a download hint if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Kaggle: search 'Flight Price Dataset Bangladesh'."
        )

    df = pd.read_csv(path)
    logger.info("Loaded %s rows × %d columns from '%s'", f"{len(df):,}", df.shape[1], path.name)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Log a structured inspection report: shape, dtypes, nulls, and descriptive statistics.
    """
    sep = "=" * 60

    logger.info("\n%s\nDATASET SHAPE\n%s\n  Rows    : %s\n  Columns : %d",
                sep, sep, f"{df.shape[0]:,}", df.shape[1])

    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    dtype_info  = pd.DataFrame({
        "dtype":      df.dtypes.astype(str),
        "null_count": null_counts,
        "null_%":     null_pct,
        "unique":     df.nunique(),
    })
    logger.info("\n%s\nCOLUMN DTYPES & NULL COUNTS\n%s\n%s",
                sep, sep, dtype_info.to_string())

    logger.info("\n%s\nSAMPLE ROWS (head 5)\n%s\n%s",
                sep, sep, df.head().to_string())

    logger.info("\n%s\nDESCRIPTIVE STATISTICS (numeric)\n%s\n%s",
                sep, sep, df.describe().T.to_string())

    date_col  = df["Date"].dtype if "Date" in df.columns else "N/A"
    fare_col  = df.get("Base Fare (BDT)", df.get("Base_Fare", pd.Series([0])))
    observations = [
        f"  • {df.duplicated().sum()} fully duplicate rows detected.",
        f"  • Columns with nulls: {list(null_counts[null_counts > 0].index) or 'None'}",
        f"  • Date column dtype: {date_col}  (needs datetime conversion if 'object')",
        f"  • Numeric fare range: BDT {fare_col.min():,.0f} – {fare_col.max():,.0f}",
    ]
    logger.info("\n%s\nINITIAL OBSERVATIONS\n%s\n%s",
                sep, sep, "\n".join(observations))
