"""
scripts/extract_flow.py

Batch Optical Flow Extraction
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from data.optical_flow import process_dataset


def main(args):

    print("\n===== OPTICAL FLOW EXTRACTION =====")

    print(f"[INFO] Input Dataset : {args.input_dir}")
    print(f"[INFO] Output Folder : {args.output_dir}")

    process_dataset(
        dataset_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    print("\n[INFO] Extraction Complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract Optical Flow from Dataset"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="UCF101 dataset root",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Flow output directory",
    )

    args = parser.parse_args()

    main(args)