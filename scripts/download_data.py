"""Download and prepare the UCI Adult Census Income dataset as data/raw.csv."""
import io
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
import numpy as np

UCI_ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

def download_and_process():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "raw.csv"

    print(f"Downloading Adult dataset from {UCI_ADULT_URL}...")
    req = urllib.request.Request(
        UCI_ADULT_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Load adult.data
        data_text = z.read("adult.data").decode("utf-8")
        df_train = pd.read_csv(
            io.StringIO(data_text),
            header=None,
            names=COLUMN_NAMES,
            skipinitialspace=True,
            na_values=["?", ""],
        )

        # Load adult.test
        test_text = z.read("adult.test").decode("utf-8")
        # adult.test has an extra header line: "|1x3 Cross validator"
        lines = [line for line in test_text.splitlines() if not line.startswith("|") and line.strip()]
        df_test = pd.read_csv(
            io.StringIO("\n".join(lines)),
            header=None,
            names=COLUMN_NAMES,
            skipinitialspace=True,
            na_values=["?", ""],
        )

    # Normalize income target column (test targets have a trailing dot, e.g. '<=50K.')
    df_train["income"] = df_train["income"].astype(str).str.strip().str.rstrip(".")
    df_test["income"] = df_test["income"].astype(str).str.strip().str.rstrip(".")

    # Combine both sets into the complete standard 48,842-record dataset
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop any trailing empty/null target rows if present
    df = df[df["income"].isin([">50K", "<=50K"])].copy()

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col] == "nan", col] = np.nan

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['income'].value_counts(normalize=True)}")
    print(f"Sex distribution:\n{df['sex'].value_counts(normalize=True)}")
    print(f"Race distribution:\n{df['race'].value_counts(normalize=True)}")

    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path.resolve()}")

if __name__ == "__main__":
    download_and_process()
