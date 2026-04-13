"""CSV transformation and writing logic.

Data transformation and CSV file appending — no UI, no user interaction.
Converts reviewed JSON data into tabular CSV rows and appends them to disk.
"""

import csv
from typing import Any

from gpcr_tools.config import AUX_PROTEIN_DISPATCH, CSV_SCHEMA, get_config


def sanitize_value(value: Any) -> str:
    """Convert a value to a clean string for CSV output."""
    if value is None:
        return ""
    return str(value).strip()


def transform_for_csv(pdb_id: str, data: dict) -> dict[str, list[dict[str, str]]]:
    """Transform reviewed PDB data into CSV-ready row dictionaries.

    Applies scientific transformations via :mod:`logic`:
    * Multi-chain receptor truncation to primary protomer
    * Orphaned-ligand radar (warns when ligands sit on truncated chains)
    * ``label_asym_id`` mapping (auth_asym_id → PDB standard identifiers)
    * Structure note enrichment with oligomer annotations

    Returns a mapping of CSV filename → list of row dicts.
    """
    from gpcr_tools.csv_generator.logic import (
        apply_db_truncation,
        build_structure_note,
        collect_ligand_chains,
        map_label_asym_id,
    )

    rows_map: dict[str, list[dict[str, str]]] = {fname: [] for fname in CSV_SCHEMA}

    s_info = data.get("structure_info") or {}
    r_info = data.get("receptor_info") or {}
    oligo = data.get("oligomer_analysis") or {}
    label_map = oligo.get("label_asym_id_map") or {}

    receptor_chain = sanitize_value(r_info.get("chain_id"))
    receptor_uniprot = sanitize_value(r_info.get("uniprot_entry_name"))

    # ── Truncation + orphaned-ligand radar ─────────────────────────
    ligand_chains = collect_ligand_chains(data.get("ligands") or [])
    receptor_chain, receptor_uniprot, truncation_note = apply_db_truncation(
        receptor_chain,
        receptor_uniprot,
        oligo,
        ligand_chains,
    )

    # ── Structure note enrichment ──────────────────────────────────
    s_note = build_structure_note(s_info, oligo, truncation_note)

    # ── structures.csv ─────────────────────────────────────────────
    rows_map["structures.csv"].append(
        {
            "PDB": pdb_id,
            "Receptor_UniProt": receptor_uniprot,
            "Method": sanitize_value(s_info.get("method")),
            "Resolution": sanitize_value(s_info.get("resolution")),
            "State": sanitize_value((s_info.get("state") or {}).get("value") or "").capitalize(),
            "ChainID": receptor_chain,
            "label_asym_id": map_label_asym_id(receptor_chain, label_map),
            "Note": s_note,
            "Date": sanitize_value(s_info.get("release_date")),
        }
    )

    # ── ligands.csv ────────────────────────────────────────────────
    for lig in data.get("ligands") or []:
        smiles = lig.get("SMILES_stereo") or lig.get("SMILES") or ""
        lig_chain = sanitize_value(lig.get("chain_id"))
        rows_map["ligands.csv"].append(
            {
                "PDB": pdb_id,
                "ChainID": lig_chain,
                "label_asym_id": map_label_asym_id(lig_chain, label_map),
                "Name": sanitize_value(lig.get("name")),
                "PubChemID": sanitize_value(lig.get("pubchem_id")),
                "Role": sanitize_value((lig.get("role") or {}).get("value")),
                "Title": sanitize_value(lig.get("name")),
                "Type": sanitize_value(lig.get("type")),
                "Date": sanitize_value(s_info.get("release_date")),
                "In structure": "",
                "SMILES": sanitize_value(smiles),
                "InChIKey": sanitize_value(lig.get("InChIKey")),
                "Sequence": sanitize_value(lig.get("Sequence")),
            }
        )

    # ── g_proteins.csv ─────────────────────────────────────────────
    partners = data.get("signaling_partners") or {}
    if partners.get("g_protein"):
        gp = partners["g_protein"]
        alpha_chain = sanitize_value((gp.get("alpha_subunit") or {}).get("chain_id"))
        beta_chain = sanitize_value((gp.get("beta_subunit") or {}).get("chain_id"))
        gamma_chain = sanitize_value((gp.get("gamma_subunit") or {}).get("chain_id"))
        rows_map["g_proteins.csv"].append(
            {
                "PDB": pdb_id,
                "Alpha_UniProt": sanitize_value(
                    (gp.get("alpha_subunit") or {}).get("uniprot_entry_name")
                ),
                "Alpha_ChainID": alpha_chain,
                "Alpha_label_asym_id": map_label_asym_id(alpha_chain, label_map),
                "Beta_UniProt": sanitize_value(
                    (gp.get("beta_subunit") or {}).get("uniprot_entry_name")
                ),
                "Beta_ChainID": beta_chain,
                "Beta_label_asym_id": map_label_asym_id(beta_chain, label_map),
                "Gamma_UniProt": sanitize_value(
                    (gp.get("gamma_subunit") or {}).get("uniprot_entry_name")
                ),
                "Gamma_ChainID": gamma_chain,
                "Gamma_label_asym_id": map_label_asym_id(gamma_chain, label_map),
                "Note": sanitize_value(gp.get("note")),
            }
        )

    # ── arrestins.csv ──────────────────────────────────────────────
    if partners.get("arrestin"):
        ar = partners["arrestin"]
        ar_chain = sanitize_value(ar.get("chain_id"))
        rows_map["arrestins.csv"].append(
            {
                "PDB": pdb_id,
                "UniProt": sanitize_value(ar.get("uniprot_entry_name")),
                "ChainID": ar_chain,
                "label_asym_id": map_label_asym_id(ar_chain, label_map),
                "Note": sanitize_value(ar.get("note")),
            }
        )

    # ── auxiliary protein CSVs ─────────────────────────────────────
    for aux in data.get("auxiliary_proteins") or []:
        target = AUX_PROTEIN_DISPATCH.get(
            (aux.get("type") or {}).get("value") or "Other",
            "other_aux_proteins.csv",
        )
        rows_map[target].append({"PDB": pdb_id, "Name": sanitize_value(aux.get("name"))})

    return rows_map


def append_to_csvs(csv_data_map: dict[str, list[dict[str, str]]]) -> None:
    """Append rows to the appropriate CSV files, creating them with headers if needed.

    Performs a header migration check: if an existing file has outdated headers
    (e.g. missing ``label_asym_id`` columns), a CsvSchemaMismatchError is raised
    to prevent silent column misalignment.
    """
    from gpcr_tools.csv_generator.exceptions import CsvSchemaMismatchError

    cfg = get_config()
    csv_dir = cfg.csv_output_dir
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: validate all schemas before writing anything to avoid
    # partial writes (e.g. structures.csv written but ligands.csv rejected).
    for filename, rows in csv_data_map.items():
        if not rows:
            continue
        filepath = csv_dir / filename
        expected_fields = CSV_SCHEMA[filename]
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                existing_header = f.readline().strip().split("\t")
            if existing_header != list(expected_fields):
                raise CsvSchemaMismatchError(
                    filename=filename,
                    expected_fields=expected_fields,
                    found_fields=existing_header,
                )

    for filename, rows in csv_data_map.items():
        if not rows:
            continue
        filepath = csv_dir / filename
        expected_fields = CSV_SCHEMA[filename]
        exists = filepath.exists()
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=expected_fields, delimiter="\t")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
