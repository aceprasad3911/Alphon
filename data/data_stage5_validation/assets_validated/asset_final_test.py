import csv
import sys
from pathlib import Path
from collections import defaultdict

# --- File Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "enrichment" / "isin" / "all_assets_final.csv"
OUTPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "enrichment" / "isin" / "all_assets_final_enriched.csv"
FAILED_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "enrichment" / "isin" / "failed_inception_dates.csv"


def validate_csv(input_file, output_file, failed_file):
    errors = []
    total_rows = 0
    invalid_rows = set()
    issues_by_row = defaultdict(list)

    valid_rows = []
    failed_rows = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            print("❌ No headers found in CSV file.")
            sys.exit(1)

        for row_num, row in enumerate(reader, start=2):  # 2 = header offset
            total_rows += 1
            ticker = row.get('ticker', '').strip()

            # --- 1. Field completeness check ---
            for field in row:
                value = str(row[field]).strip()
                if field != 'isin' and value in ['', 'None', 'N/A']:
                    msg = f"Missing or invalid value in '{field}'"
                    errors.append(f"Row {row_num} ({ticker}): {msg}")
                    issues_by_row[row_num].append(msg)
                    invalid_rows.add(row_num)

            # --- 2. active_status consistency check ---
            active_status = str(row.get('active_status', '')).strip().lower()
            end_date = str(row.get('end_date', '')).strip()
            notes = str(row.get('notes', '')).strip()

            if active_status == 'false':
                if end_date in ['', 'None', 'N/A']:
                    msg = "'active_status=False' but 'end_date' is missing"
                    errors.append(f"Row {row_num} ({ticker}): {msg}")
                    issues_by_row[row_num].append(msg)
                    invalid_rows.add(row_num)
                if notes in ['', 'None', 'N/A']:
                    msg = "'active_status=False' but 'notes' are missing"
                    errors.append(f"Row {row_num} ({ticker}): {msg}")
                    issues_by_row[row_num].append(msg)
                    invalid_rows.add(row_num)

            # --- 3. Store rows based on result ---
            if row_num in invalid_rows:
                failed_rows.append(row)
            else:
                valid_rows.append(row)

    # --- 4. Output Results ---
    print("=" * 75)
    print(f"📊 CSV Validation Report for: {input_file}")
    print("=" * 75)
    print(f"Total rows checked: {total_rows}")
    print(f"Rows with issues:   {len(invalid_rows)}")
    print(f"Total issues found: {len(errors)}")

    if total_rows > 0:
        valid_percentage = round(100 * (total_rows - len(invalid_rows)) / total_rows, 2)
        print(f"Data validity rate: {valid_percentage}%")
    print("=" * 75)

    # --- 5. Write valid and failed CSVs ---
    if headers:
        if valid_rows:
            with open(output_file, 'w', newline='', encoding='utf-8') as vf:
                writer = csv.DictWriter(vf, fieldnames=headers)
                writer.writeheader()
                writer.writerows(valid_rows)

        if failed_rows:
            with open(failed_file, 'w', newline='', encoding='utf-8') as ff:
                writer = csv.DictWriter(ff, fieldnames=headers)
                writer.writeheader()
                writer.writerows(failed_rows)

    # --- 6. Print detailed issues ---
    if errors:
        print("\n❌ Detailed Issues:")
        for e in errors:
            print(" -", e)
        print(f"\n❗ Validation failed. {len(failed_rows)} rows written to: {failed_file}")
        print(f"✅ {len(valid_rows)} valid rows written to: {output_file}")
        sys.exit(1)
    else:
        print("\n✅ All checks passed successfully!")
        print(f"✅ Full validated dataset saved to: {output_file}")
        sys.exit(0)


if __name__ == "__main__":
    print("🔍 Running CSV validation...\n")
    validate_csv(INPUT_CSV, OUTPUT_CSV, FAILED_CSV)
