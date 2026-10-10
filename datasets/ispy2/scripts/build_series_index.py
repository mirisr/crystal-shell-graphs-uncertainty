"""Query NBIA REST (via tcia_utils) for every series in the ISPY2 and
ACRIN-6698 collections and write a per-series index CSV under ../metadata/.

Re-run this whenever TCIA updates the collection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

HERE = Path(__file__).resolve().parent
META = HERE.parent / "metadata"


def dump_collection(collection: str, out_path: Path) -> pd.DataFrame:
    print(f"[{collection}] querying NBIA getSeries...")
    series = nbia.getSeries(collection=collection)
    if not series:
        raise SystemExit(f"NBIA returned no series for collection {collection!r}")
    df = pd.DataFrame(series)
    df.to_csv(out_path, index=False)
    print(
        f"[{collection}] wrote {out_path.relative_to(HERE.parent.parent.parent)} "
        f"({len(df):,} rows, {df['FileSize'].sum() / 1e12:.2f} TB total, "
        f"{df['PatientID'].nunique()} patients)"
    )
    return df


def main() -> int:
    META.mkdir(parents=True, exist_ok=True)
    dump_collection("ISPY2", META / "ispy2_series_index.csv")
    dump_collection("ACRIN-6698", META / "acrin6698_series_index.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
