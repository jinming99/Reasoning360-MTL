"""
Create a balanced GURU‑18K training dataset.

This script assumes you have already downloaded the full GURU RL dataset
using ``scripts/tools/download_guru.py``.  It samples 3,000 examples
from each of the six domains (math, codegen, logic, simulation,
table and stem) and writes a single Parquet file containing all
18,000 examples.  The resulting file can be passed directly to the
training scripts via the ``--train_files`` argument.

Usage::

    python scripts/tools/create_guru_18k.py --data-dir ./data/train \
        --output ./data/train/guru_18k_mix.parquet

If multiple Parquet files exist for a domain, they are concatenated
prior to sampling.  The sampling is performed without replacement.
"""

import argparse
import glob
import os
import random
from typing import List

import pandas as pd

DOMAIN_PREFIXES = {
    "math": "math__",
    "codegen": "codegen__",
    "logic": "logic__",
    "simulation": "simulation__",
    "table": "table__",
    "stem": "stem__",
}


def find_domain_files(data_dir: str, prefix: str) -> List[str]:
    """Return a list of Parquet files in ``data_dir`` whose basename starts
    with ``prefix``.
    """
    pattern = os.path.join(data_dir, f"{prefix}*.parquet")
    return sorted(glob.glob(pattern))


def load_and_sample(files: List[str], n_samples: int) -> pd.DataFrame:
    """Load all Parquet files in ``files`` and randomly sample ``n_samples``
    rows (without replacement).  If the total number of rows is less than
    ``n_samples``, the sampling will be with replacement.
    """
    df_list = [pd.read_parquet(f) for f in files]
    if not df_list:
        raise FileNotFoundError(f"No Parquet files found for pattern: {files}")
    full_df = pd.concat(df_list, ignore_index=True)
    if len(full_df) >= n_samples:
        sampled_df = full_df.sample(n=n_samples, random_state=42)
    else:
        # sample with replacement if not enough examples
        sampled_df = full_df.sample(n=n_samples, replace=True, random_state=42)
    return sampled_df


def main(args: argparse.Namespace) -> None:
    domain_dfs = {}
    for domain, prefix in DOMAIN_PREFIXES.items():
        files = find_domain_files(args.data_dir, prefix)
        if not files:
            print(f"Warning: No files found for domain '{domain}' with prefix '{prefix}*'")
            continue
        print(f"Loading {len(files)} file(s) for domain '{domain}'...")
        df = load_and_sample(files, args.samples_per_domain)
        df["mtl_task"] = domain  # annotate each sample with its domain label
        domain_dfs[domain] = df
        print(f"Sampled {len(df)} rows for domain '{domain}'")

    # Combine all domains into a single DataFrame
    if not domain_dfs:
        raise ValueError("No data was loaded. Please check your data directory and domain prefixes.")
    
    # Convert all columns to string to avoid mixed type issues
    for domain, df in domain_dfs.items():
        for col in df.columns:
            try:
                # First try to convert to string
                df[col] = df[col].astype(str)
                # If the column contains '[' or '{', it might be a list or dict
                if df[col].str.contains(r'[\[\]{]').any():
                    # Leave as string if it looks like JSON
                    pass
                # Try to convert back to numeric if possible
                elif pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors='coerce')):
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' for domain '{domain}': {e}")
                df[col] = df[col].astype(str)
    
    # Combine all domains
    mixed_df = pd.concat(list(domain_dfs.values()), ignore_index=True)
    
    # Ensure all columns are string type to avoid mixed type issues
    for col in mixed_df.columns:
        if mixed_df[col].dtype == 'object':
            mixed_df[col] = mixed_df[col].astype(str)
    
    # Save the mixed dataset
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    try:
        mixed_df.to_parquet(args.output, index=False)
        print(f"Saved mixed dataset with {len(mixed_df)} rows to {args.output}")
    except Exception as e:
        print(f"Error saving mixed dataset: {e}")
        print("Troubleshooting info:")
        print(f"- Mixed DataFrame shape: {mixed_df.shape}")
        print(f"- Columns: {mixed_df.columns.tolist()}")
        print(f"- dtypes: {mixed_df.dtypes}")
        raise
    
    # Optionally write one Parquet file per domain
    if args.separate_dir:
        os.makedirs(args.separate_dir, exist_ok=True)
        for domain, df in domain_dfs.items():
            out_path = os.path.join(args.separate_dir, f"{domain}.parquet")
            try:
                df.to_parquet(out_path, index=False)
                print(f"Saved {domain}: {len(df)} rows → {out_path}")
            except Exception as e:
                print(f"Error saving {domain} dataset: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GURU-18K dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the downloaded GURU train Parquet files",
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=3000,
        help="Number of samples to draw per domain (default: 3000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the mixed Parquet file",
    )
    parser.add_argument(
        "--separate-dir",
        type=str,
        default=None,
        help="If set, write <separate-dir>/<domain>.parquet for every domain.",
    )

    main(parser.parse_args())
