import sys
import time
import requests
from lxml import etree

URL_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pubmed&id={}&retmode=xml"
)


def get_mesh_for_pmid(pmid):
    """
    Query PubMed eFetch for a given PMID, parse out MeSH terms.
    Returns a list of MeSH headings, or empty list if none found.
    """
    url = URL_EFETCH.format(pmid)
    resp = requests.get(url)
    if resp.status_code != 200:
        return []

    root = etree.fromstring(resp.content)
    # Mesh headings appear under <MeshHeadingList><MeshHeading><DescriptorName>...</DescriptorName>
    mesh_terms = root.xpath("//MeshHeadingList/MeshHeading/DescriptorName/text()")
    return [term.strip() for term in mesh_terms if term.strip()]


def pmids_to_mesh(input_pmids_txt, output_mesh_txt):
    """
    Reads each PMID from <input_pmids_txt> (ignoring lines that start with NOT_FOUND),
    queries eFetch, then writes 'PMID\tmesh_term1;mesh_term2;...' to <output_mesh_txt>.
    """
    with open(input_pmids_txt, "r", encoding="utf-8") as f_in, open(
        output_mesh_txt, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            line = line.strip()
            if not line or line.startswith("NOT_FOUND"):
                continue

            parts = line.split("\t", maxsplit=1)
            pmid = parts[0].strip()

            if pmid == "NOT_FOUND":
                continue

            time.sleep(0.5) # be nice to the server
            mesh_terms = get_mesh_for_pmid(pmid)
            if mesh_terms:
                mesh_str = "; ".join(mesh_terms)
                f_out.write(f"{pmid}\t{mesh_str}\n")
                print(f"[OK] PMID {pmid} => {len(mesh_terms)} MeSH terms")
            else:
                f_out.write(f"{pmid}\tNO_MESH_TERMS\n")
                print(f"[WARN] PMID {pmid} => no MeSH terms found")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pmids_to_mesh.py <pmids_in.txt> <mesh_out.txt>")
        sys.exit(1)

    input_pmids = sys.argv[1]  # plos/train_pmids.txt, plos/valid_pmids.txt, plos/test_pmids.txt
    output_mesh = sys.argv[2]  # plos/train_mesh.txt, plos/valid_mesh.txt, plos/test_mesh.txt
    pmids_to_mesh(input_pmids, output_mesh)
