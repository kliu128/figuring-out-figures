#!/usr/bin/env python
# coding: utf-8

# In[13]:


import json
from pathlib import Path
import ssw
from typing import List, Literal
import re
import multiprocessing as mp

# In[19]:


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:'\"-()"
repl_pattern = r"[^a-zA-Z0-9\.\,\;\:\'\"\-\(\) ]"
aligner = ssw.Aligner()
aligner.matrix = ssw.ScoreMatrix(alphabet=alphabet)


def find_references(text: str, fig_num: str, caption_blacklist: str, type: Literal["figure", "table"], window_size: int = 200) -> List[str]:
    """
    Finds references to Figure (fig_num) in a provided text corpus.

    Parameters
    ----------
    text : str
        Text corpus
    fig_num : int
        Figure number to look for
    window_size : int
        Number of characters before and after the match to include in the
        context window
    """
    contexts = []

    fig_pattern = rf"(?i)((?:Fig\.?|Figure) {fig_num})"
    tab_pattern = rf"(?i)((?:Table|Tab\.?) {fig_num})"
    regex = fig_pattern if type == "figure" else tab_pattern

    alignment = aligner.align(
        reference=re.sub(repl_pattern, "X", text.lower()),
        query=re.sub(repl_pattern, "X", caption_blacklist.lower()))
    start, end = alignment.reference_begin, alignment.reference_end + 1
    print("Looking for", type, fig_num)
    print("Deleting caption", text[start:end], "at", start, end)
    print("Alignment Report:")
    print(alignment.alignment_report())
    text = text[:start] + "X" * (end - start) + text[end:]
    for match in re.finditer(regex, text):
        start, end = match.span()
        window = text[start - window_size:end + window_size]
        contexts.append(window)
    print("Found", len(contexts), "contexts")
    return contexts

#import pprint
# pprint.pprint(find_references(pdf, 1, "FIG. 1: (a) T-dependence of the resistance of samples grown at PO2 = 10")


# In[20]:


# Converter

root = Path("../scicap_data")
scicap_metadata = Path("../scicap_data/SciCap-Caption-All/train")
text_dir = Path("/data/kevin/arxiv/fulltext")
references_dir = root / "references"

references_dir.mkdir(exist_ok=True)


def process_file(json_file: Path):
    with open(json_file) as f:
        metadata = json.load(f)

    paper_id = metadata["paper-ID"]
    fig_num = re.search(r"(?:Figure|Table)(.+)-",
                        metadata["figure-ID"]).group(1)
    fulltext_file = text_dir / f"{paper_id}.txt"

    if not fulltext_file.exists():
        return

    with fulltext_file.open() as f:
        text = f.read()

    # Find all references
    references = find_references(
        text, fig_num, metadata["0-originally-extracted"], "figure" if "Figure" in metadata["figure-ID"] else "table")

    # Write to file
    references_file = root / "references" / json_file.name
    with references_file.open("w") as f:
        json.dump({"references": references}, f)

    print("Saved Figure", fig_num, "for", paper_id, metadata["figure-ID"])
    print("fulltext", fulltext_file)
    print("reference:", references_file)
    print("json:", json_file)
    print("=" * 80)


pool = mp.Pool()
for json_file in scicap_metadata.iterdir():
    pool.apply_async(process_file, args=(json_file,))
pool.close()
pool.join()
