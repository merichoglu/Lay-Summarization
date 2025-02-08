import os
import json
import argparse


def load_pmids_titles(pmid_file):
    """
    Load PMIDs and their corresponding titles from a file.
    """
    pmid_to_title = {}
    with open(pmid_file, "r", encoding="utf-8") as f:
        for line in f:
            pmid, title = line.strip().split("\t", 1)
            pmid_to_title[pmid] = title
    return pmid_to_title


def load_mesh_terms(mesh_file):
    """
    Load MeSH terms from a file and return a dictionary mapping PMIDs to MeSH terms.
    """
    mesh_dict = {}
    with open(mesh_file, "r", encoding="utf-8") as f:
        for line in f:
            pmid, terms = line.strip().split("\t", 1)
            mesh_dict[pmid] = terms.split("; ")
    return mesh_dict


def match_titles_with_json(json_file, pmid_to_title):
    """
    Match titles in the JSON file with PMIDs from pmid_to_title and return a mapping of JSON entries to PMIDs.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    title_to_json_entry = {entry["title"]: entry for entry in data}
    matched_data = []

    for pmid, title in pmid_to_title.items():
        if title in title_to_json_entry:
            json_entry = title_to_json_entry[title]
            json_entry["pmid"] = pmid
            matched_data.append(json_entry)

    return matched_data


def merge_mesh_with_json(json_data, mesh_dict):
    """
    Add MeSH terms to the matched JSON data using the PMIDs.
    """
    for entry in json_data:
        pmid = entry.get("pmid")
        if pmid:
            entry["mesh_terms"] = mesh_dict.get(pmid, [])  # Add MeSH terms if available
    return json_data


def prepare_data(data_dir, output_dir):
    """
    Prepare the datasets by merging PMIDs and MeSH terms into JSON files and saving the updated datasets.
    """
    for split in ["train", "val", "test"]:
        json_file = os.path.join(data_dir, f"{split}.json")
        pmid_file = os.path.join(data_dir, f"{split}_pmids_noNotFound.txt")
        mesh_file = os.path.join(data_dir, f"{split}_mesh.txt")

        if (
            not os.path.exists(json_file)
            or not os.path.exists(pmid_file)
            or not os.path.exists(mesh_file)
        ):
            print(f"Skipping {split}: Missing JSON, PMIDs, or MeSH file.")
            continue

        print(f"Processing {split}...")

        pmid_to_title = load_pmids_titles(pmid_file)
        mesh_dict = load_mesh_terms(mesh_file)

        # Match titles with JSON and merge MeSH terms
        matched_json = match_titles_with_json(json_file, pmid_to_title)
        enriched_data = merge_mesh_with_json(matched_json, mesh_dict)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{split}_prepared.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=4)

        print(f"Saved prepared {split} data to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data by merging PMIDs and MeSH terms into JSON files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the raw JSON, PMIDs, and MeSH files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the prepared JSON files.",
    )
    args = parser.parse_args()

    prepare_data(args.data_dir, args.output_dir)
