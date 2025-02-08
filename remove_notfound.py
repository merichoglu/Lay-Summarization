import sys


def remove_not_found_lines(input_file, output_file):
    """
    Reads `input_file` line by line.
    If a line starts with `NOT_FOUND` (case-insensitive), skip it.
    Otherwise, write it to `output_file`.
    This was needed because some articles did not have valid PMIDs.
    """
    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # format is: "NOT_FOUND<TAB>Title..."
            if line_stripped.split("\t", 1)[0].lower() == "not_found":
                continue

            f_out.write(line)


if __name__ == "__main__":
    """
    Example usage:
      python remove_notfound.py pmids_in.txt pmids_out.txt
    """
    if len(sys.argv) != 3:
        print("Usage: python remove_notfound.py <input_pmids_file> <output_pmids_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    remove_not_found_lines(input_path, output_path)
    print(f"Done. Filtered lines from '{input_path}' and wrote to '{output_path}'.")
