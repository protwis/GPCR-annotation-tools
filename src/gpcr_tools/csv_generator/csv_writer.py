"""CSV transformation and writing logic.

Data transformation and CSV file appending — no UI, no user interaction.
Converts reviewed JSON data into tabular CSV rows and appends them to disk.
"""

import csv
import os
from typing import Any

from gpcr_tools.config import (
    AUX_PROTEIN_DISPATCH,
    CSV_SCHEMA,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_SKIPPED_APO,
    get_config,
    is_empty_key,
)


def sanitize_value(value: Any) -> str:
    """Convert a value to a clean string for CSV output."""
    if value is None:
        return ""
    return str(value).strip()


def _primary_chain(value: Any) -> str:
    """Collapse a multi-chain G-protein subunit value to a single chain.

    A single subunit physically occupies one chain per complex; a multi-value
    field (e.g. ``"C, D"``) means the asymmetric unit holds more than one
    redundant copy of the same complex. The structures row already collapses the
    receptor to its primary protomer, so the subunit collapses likewise — to the
    chain of the primary complex, which depositions list first. (A geometry-based
    pairing to the primary receptor chain would be exact; this first-listed
    heuristic covers the common ordered case.)
    """
    text = sanitize_value(value)
    # Split on comma or semicolon, tolerating optional spaces, to match the chain
    # parsers used elsewhere in the codebase; take the first (primary complex).
    return text.replace(";", ",").split(",")[0].strip()


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
        resolve_partner_protomer,
    )

    rows_map: dict[str, list[dict[str, str]]] = {fname: [] for fname in CSV_SCHEMA}

    s_info = data.get("structure_info") or {}
    r_info = data.get("receptor_info") or {}
    oligo = data.get("oligomer_analysis") or {}
    label_map = oligo.get("label_asym_id_map") or {}
    nonpolymer_instances = oligo.get("nonpolymer_instance_index") or {}

    receptor_chain = sanitize_value(r_info.get("chain_id"))
    receptor_uniprot = sanitize_value(r_info.get("uniprot_entry_name"))
    # NOTE: receptor_info.oligomeric_state is captured in the aggregated JSON but is
    # intentionally NOT yet exported to any CSV column or structure note. Adding it
    # requires the four-place CSV schema lockstep (CSV_SCHEMA in config.py,
    # transform_for_csv here, test_csv_writer.py, and the README output table); it is
    # deferred to a dedicated change so this stays a minimal addition.

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

    # ── Dimer partner protomer (recorded, not dropped) ─────────────
    partner_uniprot, partner_chain = resolve_partner_protomer(oligo, receptor_chain)

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
            "Partner_UniProt": partner_uniprot,
            "Partner_ChainID": partner_chain,
            "Note": s_note,
            "Date": sanitize_value(s_info.get("release_date")),
        }
    )

    # ── ligands.csv ────────────────────────────────────────────────
    for lig in data.get("ligands") or []:
        if not isinstance(lig, dict):
            continue
        # Fail-safe: a ligand the validator could not find in the structure
        # (GHOST_LIGAND) is left out of the export unless a curator explicitly
        # confirmed it.  The model sometimes annotates a ligand the paper
        # discusses but that this deposition does not actually model; writing it
        # would record an interaction for a molecule absent from the structure.
        if lig.get("validation_status") == VALIDATION_GHOST_LIGAND and not lig.get(
            "curator_kept_ghost"
        ):
            continue
        # Skip an apo / "no ligand" placeholder: the model sometimes emits a row
        # to note the structure also has a ligand-free form. It is not a bound
        # ligand, so it must not become a ligand-interaction row (it would
        # otherwise carry a spurious binding site / role). The validator tags
        # these SKIPPED_APO; the value checks also catch any that slipped tagging.
        if (
            lig.get("validation_status") == VALIDATION_SKIPPED_APO
            or sanitize_value(lig.get("type")) == "none"
            or sanitize_value(lig.get("name")) == "Apo"
            or sanitize_value((lig.get("role") or {}).get("value")) == "Apo (no ligand)"
        ):
            continue
        smiles = lig.get("SMILES_stereo") or lig.get("SMILES") or ""
        lig_chain = sanitize_value(lig.get("chain_id"))
        # A non-polymer ligand's label_asym_id is its OWN mmCIF instance label(s).
        # Never route it through the polymer label_asym_id_map (which covers
        # protein chains only) — mapping the ligand's auth chain through that map
        # stamps the receptor's chain label onto the ligand row. One modelled
        # copy -> its label; several -> all of them, comma-joined (mirroring the
        # ChainID column). Unindexed ligand -> blank (no protein-chain fallback).
        comp_id = sanitize_value(lig.get("chem_comp_id"))
        instances = nonpolymer_instances.get(comp_id) if comp_id else None
        if instances:
            lig_label = ", ".join(
                sanitize_value(i.get("label_asym_id")) for i in instances if i.get("label_asym_id")
            )
        else:
            lig_label = ""
        rows_map["ligands.csv"].append(
            {
                "PDB": pdb_id,
                "ChainID": lig_chain,
                "label_asym_id": lig_label,
                "Name": sanitize_value(lig.get("name")),
                # The schema tells the model to emit the string "None" when there
                # is no PubChem id; normalize that sentinel to empty rather than
                # writing a literal "None" into the numeric column.
                "PubChemID": ""
                if is_empty_key(lig.get("pubchem_id"))
                else sanitize_value(lig.get("pubchem_id")),
                "Role": sanitize_value((lig.get("role") or {}).get("value")),
                # A dual-role ligand the model split per site carries a site_ref
                # (e.g. orthosteric / allosteric); blank for an ordinary ligand.
                # This keeps the per-site rows distinct rather than duplicate.
                "Site": sanitize_value(lig.get("site_ref")),
                "Title": sanitize_value(lig.get("name")),
                "Type": sanitize_value(lig.get("type")),
                "Date": sanitize_value(s_info.get("release_date")),
                "In structure": "",
                "SMILES": sanitize_value(smiles),
                "InChIKey": sanitize_value(lig.get("InChIKey")),
                "Sequence": sanitize_value(lig.get("Sequence")),
                "is_endogenous": sanitize_value(lig.get("is_endogenous")),
            }
        )

    # ── g_proteins.csv ─────────────────────────────────────────────
    partners = data.get("signaling_partners") or {}
    if partners.get("g_protein"):
        gp = partners["g_protein"]
        alpha_chain = _primary_chain((gp.get("alpha_subunit") or {}).get("chain_id"))
        beta_chain = _primary_chain((gp.get("beta_subunit") or {}).get("chain_id"))
        gamma_chain = _primary_chain((gp.get("gamma_subunit") or {}).get("chain_id"))
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

    A file with no rows for this batch is still created header-only, so the
    downstream build never hits a missing file (e.g. grk/ramp when a batch has no
    such entities). Files use LF line endings to match the consumed data.

    Performs a header migration check: if an existing file has outdated headers
    (e.g. missing ``label_asym_id`` columns), a CsvSchemaMismatchError is raised
    to prevent silent column misalignment.
    """
    from gpcr_tools.csv_generator.exceptions import CsvSchemaMismatchError

    cfg = get_config()
    csv_dir = cfg.csv_output_dir
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: validate the schema of every existing target file before
    # writing anything, to avoid partial writes (e.g. structures.csv written but
    # ligands.csv rejected). Checked even for empty inputs so a stale header is
    # caught rather than silently left behind.
    for filename in csv_data_map:
        filepath = csv_dir / filename
        if not filepath.exists():
            continue
        expected_fields = CSV_SCHEMA[filename]
        with open(filepath, encoding="utf-8") as f:
            existing_header = f.readline().strip().split("\t")
        if existing_header != list(expected_fields):
            raise CsvSchemaMismatchError(
                filename=filename,
                expected_fields=expected_fields,
                found_fields=existing_header,
            )

    for filename, rows in csv_data_map.items():
        filepath = csv_dir / filename
        expected_fields = CSV_SCHEMA[filename]

        # No rows for this file: still ensure a header-only file exists so the
        # downstream build never sees a missing file. Existing files are left
        # untouched (don't clobber data written for another PDB in this run).
        if not rows:
            if not filepath.exists():
                tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f, fieldnames=expected_fields, delimiter="\t", lineterminator="\n"
                    ).writeheader()
                os.replace(tmp_path, filepath)
            continue

        pdb_col = expected_fields[0]  # first column is the PDB id in every schema
        incoming_pdbs = {r.get(pdb_col) for r in rows}

        # Upsert, not blind append: drop any existing rows for the same PDB(s)
        # before writing, so re-curating a PDB replaces its rows instead of
        # accumulating duplicate, conflicting entries. Rewrite atomically.
        kept: list[dict[str, str]] = []
        if filepath.exists():
            with open(filepath, newline="", encoding="utf-8") as f:
                kept = [
                    row
                    for row in csv.DictReader(f, delimiter="\t")
                    if row.get(pdb_col) not in incoming_pdbs
                ]

        tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=expected_fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(kept)
            writer.writerows(rows)
        os.replace(tmp_path, filepath)
