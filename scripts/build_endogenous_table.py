"""Build the GtoPdb endogenous-ligand lookup (a flat InChIKey + PubChem CID set).

``is_endogenous`` is a ligand-intrinsic property: is this compound an endogenous
ligand of *any* target in the Guide to PHARMACOLOGY? -- not tied to the receptor a
particular structure happens to show it in. So the artifact is a flat set, not a
per-receptor map.

Downloads the endogenous ligand-target pairings and the ligand table, takes every
ligand that appears as an endogenous ligand, and writes its InChIKey / PubChem CID
to a gzipped JSON set shipped at ``src/gpcr_tools/data/gtopdb_endogenous_ligands.json.gz``.

Source: IUPHAR/BPS Guide to PHARMACOLOGY (https://www.guidetopharmacology.org),
licensed CC-BY-SA 4.0 -- the derived table is redistributed under the same licence
(see the NOTICE recorded in the artifact header and the repo).

Run from the repo root:  python3 scripts/build_endogenous_table.py
Re-run on each GtoPdb release to refresh the bundled artifact.
"""

import csv
import gzip
import io
import json
import urllib.request

PAIRINGS_URL = "https://www.guidetopharmacology.org/DATA/endogenous_ligand_pairings_all.csv"
LIGANDS_URL = "https://www.guidetopharmacology.org/DATA/ligands.csv"
OUT = "src/gpcr_tools/data/gtopdb_endogenous_ligands.json.gz"


def read_csv(url: str) -> tuple[str, list[dict]]:
    """Fetch a GtoPdb CSV; return (version_string, rows). The first line is a
    ``# GtoPdb Version: ...`` comment, the column header is the next line."""
    with urllib.request.urlopen(url, timeout=180) as resp:  # trusted GtoPdb host
        text = resp.read().decode("utf-8", "replace")
    version = ""
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip('"').startswith("#"):
            if "Version:" in line:
                version = line.split("Version:", 1)[1].strip().strip('"').strip()
            continue
        data_lines.append(line)
    rows = list(csv.DictReader(io.StringIO("\n".join(data_lines))))
    return version, rows


def main() -> None:
    pv, pairings = read_csv(PAIRINGS_URL)
    lv, ligands = read_csv(LIGANDS_URL)

    endogenous_ids = {r["Ligand ID"] for r in pairings if r.get("Ligand ID")}

    inchikeys: set[str] = set()
    pubchem_cids: set[str] = set()
    for row in ligands:
        if row.get("Ligand ID") not in endogenous_ids:
            continue
        ik = (row.get("InChIKey") or "").strip()
        cid = (row.get("PubChem CID") or "").strip()
        if ik:
            inchikeys.add(ik)
        if cid:
            pubchem_cids.add(cid)

    table = {
        "source": "IUPHAR/BPS Guide to PHARMACOLOGY (guidetopharmacology.org), CC-BY-SA 4.0",
        "gtopdb_version": pv or lv,
        "n_endogenous_ligands": len(endogenous_ids),
        "inchikeys": sorted(inchikeys),
        "pubchem_cids": sorted(pubchem_cids),
    }
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(table, f, separators=(",", ":"))
    print(
        f"GtoPdb {table['gtopdb_version']}: {len(endogenous_ids)} endogenous ligands "
        f"-> {len(inchikeys)} InChIKeys, {len(pubchem_cids)} PubChem CIDs -> {OUT}"
    )


if __name__ == "__main__":
    main()
