"""CSV transformation and writing logic.

Pure data transformation — no UI, no user interaction.
Converts reviewed JSON data into tabular CSV format.
"""

import csv
from typing import Any

from gpcr_tools.config import AUX_PROTEIN_DISPATCH, CSV_SCHEMA, OUTPUT_DIR


def sanitize_value(value: Any) -> str:
    """Convert a value to a clean string for CSV output."""
    if value is None:
        return ""
    return str(value).strip()


def transform_for_csv(pdb_id: str, data: dict) -> dict[str, list[dict[str, str]]]:
    """Transform reviewed PDB data into CSV-ready row dictionaries.

    Returns a mapping of CSV filename → list of row dicts.
    """
    rows_map: dict[str, list[dict[str, str]]] = {fname: [] for fname in CSV_SCHEMA}

    s_info = data.get("structure_info", {})
    r_info = data.get("receptor_info", {})

    # ── structures.csv ──────────────────────────────────────────────
    rows_map["structures.csv"].append(
        {
            "PDB": pdb_id,
            "Receptor_UniProt": sanitize_value(r_info.get("uniprot_entry_name")),
            "Method": sanitize_value(s_info.get("method")),
            "Resolution": sanitize_value(s_info.get("resolution")),
            "State": sanitize_value(s_info.get("state", {}).get("value", "")).capitalize(),
            "ChainID": sanitize_value(r_info.get("chain_id")),
            "Note": sanitize_value(s_info.get("note")),
            "Date": sanitize_value(s_info.get("release_date")),
        }
    )

    # ── ligands.csv ─────────────────────────────────────────────────
    for lig in data.get("ligands", []):
        smiles = lig.get("SMILES_stereo") or lig.get("SMILES", "")
        rows_map["ligands.csv"].append(
            {
                "PDB": pdb_id,
                "ChainID": sanitize_value(lig.get("chain_id")),
                "Name": sanitize_value(lig.get("name")),
                "PubChemID": sanitize_value(lig.get("pubchem_id")),
                "Role": sanitize_value(lig.get("role", {}).get("value")),
                "Title": sanitize_value(lig.get("name")),
                "Type": sanitize_value(lig.get("type")),
                "Date": sanitize_value(s_info.get("release_date")),
                "In structure": "",
                "SMILES": sanitize_value(smiles),
                "InChIKey": sanitize_value(lig.get("InChIKey")),
                "Sequence": sanitize_value(lig.get("Sequence")),
            }
        )

    # ── g_proteins.csv ──────────────────────────────────────────────
    partners = data.get("signaling_partners", {})
    if partners.get("g_protein"):
        gp = partners["g_protein"]
        rows_map["g_proteins.csv"].append(
            {
                "PDB": pdb_id,
                "Alpha_UniProt": sanitize_value(
                    gp.get("alpha_subunit", {}).get("uniprot_entry_name")
                ),
                "Alpha_ChainID": sanitize_value(gp.get("alpha_subunit", {}).get("chain_id")),
                "Beta_UniProt": sanitize_value(
                    gp.get("beta_subunit", {}).get("uniprot_entry_name")
                ),
                "Beta_ChainID": sanitize_value(gp.get("beta_subunit", {}).get("chain_id")),
                "Gamma_UniProt": sanitize_value(
                    gp.get("gamma_subunit", {}).get("uniprot_entry_name")
                ),
                "Gamma_ChainID": sanitize_value(gp.get("gamma_subunit", {}).get("chain_id")),
                "Note": sanitize_value(gp.get("note")),
            }
        )

    # ── arrestins.csv ───────────────────────────────────────────────
    if partners.get("arrestin"):
        ar = partners["arrestin"]
        rows_map["arrestins.csv"].append(
            {
                "PDB": pdb_id,
                "UniProt": sanitize_value(ar.get("uniprot_entry_name")),
                "ChainID": sanitize_value(ar.get("chain_id")),
                "Note": sanitize_value(ar.get("note")),
            }
        )

    # ── auxiliary protein CSVs ──────────────────────────────────────
    for aux in data.get("auxiliary_proteins", []):
        target = AUX_PROTEIN_DISPATCH.get(
            aux.get("type", {}).get("value", "Other"),
            "other_aux_proteins.csv",
        )
        rows_map[target].append({"PDB": pdb_id, "Name": sanitize_value(aux.get("name"))})

    return rows_map


def append_to_csvs(csv_data_map: dict[str, list[dict[str, str]]]) -> None:
    """Append rows to the appropriate CSV files, creating them with headers if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, rows in csv_data_map.items():
        if not rows:
            continue
        filepath = OUTPUT_DIR / filename
        exists = filepath.exists()
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_SCHEMA[filename], delimiter="\t")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
