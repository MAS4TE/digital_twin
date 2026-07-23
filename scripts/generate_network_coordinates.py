"""Regenerate the three network-size coordinate files shipped with the module.

Writes, into sourcecode/data/ (resolved relative to the digital_twin module, so
the files land where _build_network reads them):

    coordinates_small.csv    GROTTES segments only                    54 buses
    coordinates_medium.csv   TROULOUETTE + CIMETIERE BRESSOUX        106 buses
    coordinates_full.csv     verbatim copy of data/coordonnées.csv   333 buses

Source: data/coordonnées.csv, the reference building-coordinate export. It stays
in data/ (git-ignored field data) and is the ONLY input of this script; the
generated files are tracked, so a fresh clone can build any network size without
holding the raw field data.

Selection rule
--------------
A bus is kept for a given size when its (street, housenumber) falls in one of
the CABINE_STREET_SEGMENTS rows of the retained substations -- the same
segmentation _build_network applies, imported from digital_twin rather than
restated here, so the two can never drift apart:

    small  -> GROTTES              (Plateau du Bouhay, odd + even)
    medium -> TROULOUETTE          (Avenue Joseph Merlot, up to 85 / 142)
              CIMETIERE BRESSOUX   (Avenue Joseph Merlot, from 109 / 146)
    full   -> no filtering at all, a byte-for-byte copy of the source

The 3 MV passage rows of MV_PASSAGE_IDS (Paradis du Bouhay, Plantations,
Omnisport BX) are kept in ALL THREE files: _build_network extracts their
coordinates by OSM @id to place the MV backbone, which exists at every size.
The 4th passage node (Groupe Scolaire) is hard-coded in PASSAGE_COORDS and needs
no row here.

Filtering is TEXTUAL: retained lines are copied verbatim from the source, never
round-tripped through pandas. Coordinates therefore keep their exact original
representation, and `full` is guaranteed identical to the source byte for byte.

Usage
-----
    python -m scripts.generate_network_coordinates
"""
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sourcecode.digital_twin import (
    CABINE_STREET_SEGMENTS, MV_PASSAGE_IDS, NETWORK_DATA_DIR, NETWORK_SIZE_FILES,
    _process_housenumber, _process_parity,
)

SOURCE = "data/coordonnées.csv"

# size -> substations whose segments are retained. 'full' is absent on purpose:
# it is a copy, not a selection.
SIZE_SUBSTATIONS = {
    "small":  ["GROTTES"],
    "medium": ["TROULOUETTE", "CIMETIERE BRESSOUX"],
}


def _ids_by_substation(source=SOURCE):
    """Map each substation to the set of OSM @id of the buses in its segments,
    replaying the CABINE_STREET_SEGMENTS filtering of _build_network."""
    df = pd.read_csv(source, sep="\t")
    df["num"] = df["addr:housenumber"].replace("", None).apply(_process_housenumber)
    df["num_parity"] = df["num"].apply(_process_parity)
    df = df.dropna(subset=["num"])
    df_charges = df[~df["@id"].isin(MV_PASSAGE_IDS)]

    ids = {}
    for sub, street, parity, lo, hi in CABINE_STREET_SEGMENTS:
        seg = df_charges[(df_charges["addr:street"] == street)
                         & (df_charges["num_parity"] == parity)]
        if lo is not None:
            seg = seg[seg["num"] >= lo]
        if hi is not None:
            seg = seg[seg["num"] <= hi]
        ids.setdefault(sub, set()).update(seg["@id"].tolist())
    return ids


def generate(source=SOURCE, out_dir=NETWORK_DATA_DIR):
    """Write the three coordinate files; return {size: n_rows_written}."""
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Source coordinates not found ({source}). This is field data kept in "
            "the git-ignored data/ folder -- see the README for how to obtain it."
        )
    os.makedirs(out_dir, exist_ok=True)

    ids_by_sub = _ids_by_substation(source)
    passage_ids = set(MV_PASSAGE_IDS)

    with open(source, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header, body = lines[0], lines[1:]

    written = {}
    for size, subs in SIZE_SUBSTATIONS.items():
        keep = set().union(*(ids_by_sub[s] for s in subs)) | passage_ids
        out_lines = [header]
        for line in body:
            first = line.split("\t", 1)[0].strip()
            if first and int(first) in keep:
                out_lines.append(line)
        path = os.path.join(out_dir, NETWORK_SIZE_FILES[size])
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(out_lines)
        written[size] = len(out_lines) - 1

    full_path = os.path.join(out_dir, NETWORK_SIZE_FILES["full"])
    shutil.copyfile(source, full_path)
    written["full"] = len(body)
    return written


if __name__ == "__main__":
    written = generate()
    n_passage = len(MV_PASSAGE_IDS)
    for size in ("small", "medium", "full"):
        path = os.path.join(NETWORK_DATA_DIR, NETWORK_SIZE_FILES[size])
        n = written[size]
        # Rows, not buses: _build_network drops rows whose housenumber cannot be
        # parsed (2 of them in the full file), so row count >= bus count.
        print(f"{path} : {n} lignes = {n - n_passage} candidates + "
              f"{n_passage} passage MT")

    src_bytes = open(SOURCE, "rb").read()
    full_bytes = open(os.path.join(NETWORK_DATA_DIR, NETWORK_SIZE_FILES["full"]), "rb").read()
    print(f"\nfull identique a {SOURCE}, octet par octet : {src_bytes == full_bytes}")
