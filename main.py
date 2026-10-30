from __future__ import annotations

import csv
from pathlib import Path


def count_data_rows(csv_path: Path) -> int:
    """Return number of data rows (excluding header) in a CSV file."""
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header when present.
        return sum(1 for _ in reader)


def main() -> None:
    data_file = Path("Data") / "sales_data.csv"

    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return

    row_count = count_data_rows(data_file)
    print("InsightForge project is ready.")
    print(f"Loaded file: {data_file}")
    print(f"Data rows: {row_count}")


if __name__ == "__main__":
    main()
