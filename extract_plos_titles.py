import json
import sys


def extract_titles(json_path, output_txt):
    """
    Reads the PLOS JSON file and writes out one
    article title per line to output_txt.
    """
    with open(json_path, "r", encoding="utf-8") as f_in, open(
        output_txt, "w", encoding="utf-8"
    ) as f_out:
        data = json.load(f_in)
        for article in data:
            title = article.get("title", "").strip()
            if title:
                f_out.write(title + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_plos_titles.py <input_json> <output_txt>")
        sys.exit(1)

    input_json = sys.argv[1]  # plos/train.json, plos/valid.json, plos/test.json
    output_txt = sys.argv[2]  # plos/train_titles.txt, plos/valid_titles.txt, plos/test_titles.txt
    extract_titles(input_json, output_txt)
