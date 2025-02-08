import sys


def build_title_set(pmids_file):
    """
    Parses lines of the form:
      <pmid>\t<some title>
    Returns a set of the titles (the second column).
    Skips lines if pmid == NOT_FOUND (if still present).
    """
    titles_set = set()
    with open(pmids_file, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            parts = line_stripped.split("\t", 1)
            if len(parts) < 2:
                continue
            pmid, title = parts
            pmid = pmid.strip()
            if pmid.upper() == "NOT_FOUND":
                continue
            titles_set.add(title.strip())
    return titles_set


def filter_titles(titles_file, titles_out, valid_titles):
    """
    Reads lines from `titles_file` (each line is just a title).
    Keeps only those lines that appear in `valid_titles` (a set).
    Writes them to `titles_out`.
    I needed to filter out titles that were not found in the PLOS dataset to make sure the number of titles matched the number of articles with valid PMIDs.
    """
    with open(titles_file, "r", encoding="utf-8") as f_in, open(
        titles_out, "w", encoding="utf-8"
    ) as f_out:

        total_lines = 0
        kept_lines = 0

        for line in f_in:
            total_lines += 1
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # If the stripped title is in the set, a.k.a it has a valid PMID, keep it
            if line_stripped in valid_titles:
                f_out.write(line)
                kept_lines += 1

    print(f"Done filtering '{titles_file}'.")
    print(f" - {total_lines} lines read.")
    print(f" - {kept_lines} lines kept.")
    print(f" - Output => '{titles_out}'.")


def main():
    """
    Usage:
      python filter_titles.py pmids_file.txt titles_in.txt titles_out.txt
    1) Build a set of valid titles from pmids_file's second column
    2) Filter titles_in.txt to keep only lines in that set
    """
    if len(sys.argv) != 4:
        print("Usage: python filter_titles.py <pmids_file> <titles_in> <titles_out>")
        sys.exit(1)

    pmids_file = sys.argv[1]
    titles_in = sys.argv[2]
    titles_out = sys.argv[3]

    valid_titles = build_title_set(pmids_file)
    filter_titles(titles_in, titles_out, valid_titles)


if __name__ == "__main__":
    main()
