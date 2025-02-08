import sys
import time
import requests
from lxml import etree

URL_ESEARCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    '?db=pubmed&field=title&retmode=xml&term="{}"'
)


def get_pmid_from_title(title):
    """
    Query PubMed eSearch by article title.
    Returns the first PMID found or None if not found.
    """
    url = URL_ESEARCH.format(title)
    resp = requests.get(url)
    if resp.status_code != 200:
        return None

    root = etree.fromstring(resp.content)
    # PMIDs appear under <IdList><Id>some_id</Id></IdList>
    pmid_elems = root.xpath("//IdList/Id")
    if not pmid_elems:
        return None
    return pmid_elems[0].text


def titles_to_pmids(input_txt, output_txt):
    """
    Reads each title from input_txt, fetches the corresponding PMID,
    writes '<pmid>\t<title>' lines to output_txt.
    If not found, writes 'NOT_FOUND\t<title>'.
    """
    with open(input_txt, "r", encoding="utf-8") as f_in, open(
        output_txt, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            title = line.strip()
            if not title:
                continue
            time.sleep(0.5)  # be nice to the server
            pmid = get_pmid_from_title(title)
            if pmid:
                print(f"[OK] {title} => {pmid}")
                f_out.write(f"{pmid}\t{title}\n")
            else:
                print(f"[WARN] Not found: {title}")
                f_out.write(f"NOT_FOUND\t{title}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python title_to_pmids.py <titles_in.txt> <pmids_out.txt>")
        sys.exit(1)

    input_titles = sys.argv[
        1
    ]  # plos/train_titles.txt, plos/valid_titles.txt, plos/test_titles.txt
    output_pmids = sys.argv[
        2
    ]  # plos/train_pmids.txt, plos/valid_pmids.txt, plos/test_pmids.txt
    titles_to_pmids(input_titles, output_pmids)
