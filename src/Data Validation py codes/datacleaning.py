import pandas as pd
from pathlib import Path

METADATA_FILE = Path("./metadata.csv")

def final_data_validation():
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at '{METADATA_FILE}'")
        return

    print(f"Loading metadata from '{METADATA_FILE}' for final validation...")
    df = pd.read_csv(METADATA_FILE)

    print(f"\nFound {len(df)} total records to validate.")
    print("-" * 50)

    print("\n[CHECK 1] Verifying for Missing Values...")
    missing_values_count = df.isnull().sum().sum()
    if missing_values_count == 0:
        print("  \u2705 PASS: No missing (NaN) values found in any column.")
    else:
        print(f"  \u274c FAIL: Found {missing_values_count} missing values. Details below:")
        print(df.isnull().sum())

    print("\n[CHECK 2] Verifying Data Types...")
    are_types_correct = all(df.dtypes == 'object')
    if are_types_correct:
        print("  \u2705 PASS: All columns have the expected string/object data type.")
    else:
        print("  \u274c FAIL: One or more columns have an unexpected data type.")
        print(df.dtypes)

    print("\n[CHECK 3] Verifying Categorical Labels...")
    expected_classes = {'crack', 'non_crack'}
    actual_classes = set(df['class'].unique())
    expected_splits = {'train', 'validation', 'test'}
    actual_splits = set(df['split'].unique())

    if actual_classes.issubset(expected_classes) and actual_splits == expected_splits:
        print("  \u2705 PASS: 'class' and 'split' columns contain only valid, consistent values.")
    else:
        print("  \u274c FAIL: Inconsistent values found.")
        if not actual_classes.issubset(expected_classes):
            print(f"    - Unexpected class labels found: {actual_classes - expected_classes}")
        if actual_splits != expected_splits:
            print(f"    - Unexpected split labels found: {actual_splits - expected_splits}")

    print("\n[CHECK 4] Verifying for Duplicate Rows...")
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows == 0:
        print("  \u2705 PASS: No duplicate rows found in the dataset.")
    else:
        print(f"  \u274c FAIL: Found {duplicate_rows} duplicate entries that should be removed.")

    print("-" * 50)
    print("\nFinal Data Quality Report Complete.")
    if missing_values_count == 0 and are_types_correct and duplicate_rows == 0:
        print(
            "\n\u2728 VERDICT: Your dataset has passed all data cleaning and validation checks. It is accurate, consistent, and ready for training.")
    else:
        print("\n\u274c VERDICT: Issues found. Please review the failed checks above.")

if __name__ == "__main__":
    final_data_validation()