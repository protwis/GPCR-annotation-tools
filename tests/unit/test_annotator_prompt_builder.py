from gpcr_tools.annotator import prompt_builder
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_INCIDENTAL_CANDIDATE,
    DetectSignal,
)


def test_generate_chain_inventory_reminder():
    enriched_data = {
        "data": {
            "entry": {
                "polymer_entities": [
                    {
                        "rcsb_polymer_entity": {"pdbx_description": "Receptor"},
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["R"]},
                    },
                    {
                        "rcsb_polymer_entity": {"pdbx_description": "G-protein Alpha"},
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A", "B"]},
                    },
                ]
            }
        }
    }

    reminder = prompt_builder.generate_chain_inventory_reminder("7W55", enriched_data)
    assert "EXACTLY 3 unique polymer chain(s)" in reminder
    assert "Chain(s) R: Receptor" in reminder
    assert "Chain(s) A, B: G-protein Alpha" in reminder


def test_generate_chain_inventory_reminder_empty():
    reminder = prompt_builder.generate_chain_inventory_reminder("7W55", {})
    assert "contains 0 polymer chains" in reminder


def test_enhanced_simplify_pdb_json():
    enriched_data = {
        "data": {
            "entry": {
                "rcsb_id": "7W55",
                "struct": {"title": "Cool Structure"},
                "exptl": [{"method": "X-ray diffraction"}],
                "refine": [{"ls_d_res_high": 2.5}],
                "rcsb_accession_info": {"initial_release_date": "2020-01-01T00:00:00Z"},
                "polymer_entities": [
                    {
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["R"]},
                        "rcsb_polymer_entity": {"pdbx_description": "Receptor"},
                        "entity_poly": {"rcsb_entity_polymer_type": "Protein"},
                        "uniprots": [{"rcsb_id": "P12345", "gpcrdb_entry_name_slug": "rec_human"}],
                    }
                ],
                "nonpolymer_entities": [
                    {
                        "nonpolymer_comp": {
                            "chem_comp": {"id": "CLR", "name": "Cholesterol"},
                            "gpcrdb_determined_type": "small-molecule",
                            "gpcrdb_pubchem_cid": "5997",
                            "gpcrdb_pubchem_synonyms": ["Cholest-5-en-3-ol (3beta)-"],
                        },
                        "rcsb_nonpolymer_entity_container_identifiers": {"auth_asym_ids": ["C"]},
                        "rcsb_nonpolymer_entity": {"pdbx_description": "Cholesterol"},
                    },
                    {
                        "nonpolymer_comp": {
                            "chem_comp": {"id": "HOH", "name": "Water"},
                        },
                        "rcsb_nonpolymer_entity_container_identifiers": {"auth_asym_ids": ["W"]},
                    },
                ],
            }
        }
    }

    simplified = prompt_builder.enhanced_simplify_pdb_json(enriched_data)
    assert simplified["structure_details"]["pdb_id"] == "7W55"
    assert simplified["structure_details"]["method"] == "X-ray diffraction"
    assert simplified["structure_details"]["resolution"] == 2.5

    # Check polymers
    assert len(simplified["polymer_components"]) == 1
    assert simplified["polymer_components"][0]["chain_ids"] == ["R"]
    assert simplified["polymer_components"][0]["entry_names"] == ["rec_human"]

    # Check nonpolymers (HOH excluded!)
    assert len(simplified["non_polymer_components"]) == 1
    assert simplified["non_polymer_components"][0]["chem_comp_id"] == "CLR"


def test_build_prompt_parts():
    enriched_data = {"data": {"entry": {"sibling_pdbs": ["8ABC"]}}}
    parts = prompt_builder.build_prompt_parts("7W55", enriched_data, "System prompt goes here.")

    joined_parts = "".join(parts)
    assert "System prompt goes here." in joined_parts
    assert "SIBLING STRUCTURES WARNING" in joined_parts
    assert "8ABC" in joined_parts
    assert "--- PDB METADATA FOR 7W55 ---" in joined_parts
    assert "--- FULL PAPER ---" in joined_parts


def test_incidental_candidate_molecule_not_stripped():
    # Accommodate, not conceal: PLM (exclude-list AND incidental_candidate) stays visible to
    # the model; an ordinary buffer (HOH) is still stripped.
    enriched_data = {
        "data": {
            "entry": {
                "nonpolymer_entities": [
                    {"nonpolymer_comp": {"chem_comp": {"id": "PLM", "name": "Palmitic acid"}}},
                    {"nonpolymer_comp": {"chem_comp": {"id": "HOH", "name": "Water"}}},
                ]
            }
        }
    }
    simplified = prompt_builder.enhanced_simplify_pdb_json(enriched_data)
    ids = {c["chem_comp_id"] for c in simplified["non_polymer_components"]}
    assert "PLM" in ids
    assert "HOH" not in ids


def test_build_prompt_parts_zero_perturbation():
    # No signals (default / None / []) must all yield the identical prompt.
    enriched = {"data": {"entry": {}}}
    base = prompt_builder.build_prompt_parts("7W55", enriched, "P")
    none_sig = prompt_builder.build_prompt_parts("7W55", enriched, "P", detect_signals=None)
    empty_sig = prompt_builder.build_prompt_parts("7W55", enriched, "P", detect_signals=[])
    assert base == none_sig == empty_sig


def test_build_prompt_parts_injects_advisory_block_between_metadata_and_paper():
    enriched = {"data": {"entry": {}}}
    sig = DetectSignal(
        kind=SIGNAL_INCIDENTAL_CANDIDATE,
        target_ref="ligands",
        summary="PLM incidental_candidate",
        payload={"comp_id": "PLM"},
        severity=SEVERITY_ADVISORY,
    )
    joined = "".join(prompt_builder.build_prompt_parts("7W55", enriched, "P", detect_signals=[sig]))
    assert "DETECTOR EVIDENCE" in joined
    assert joined.index("PDB METADATA") < joined.index("DETECTOR EVIDENCE")
    assert joined.index("DETECTOR EVIDENCE") < joined.index("--- FULL PAPER ---")


def test_enhanced_simplify_adds_7tm_status_and_residue_length():
    # Every polymer chain gains two factual columns, listed neutrally with no
    # "this structure has X receptors" framing: a 7TM receptor reads COMPLETE,
    # a peptide ligand reads UNKNOWN 0/0.
    enriched_data = {
        "data": {
            "entry": {
                "polymer_entities": [
                    {
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["R"]},
                        "rcsb_polymer_entity": {"pdbx_description": "Receptor"},
                        "entity_poly": {
                            "rcsb_entity_polymer_type": "Protein",
                            "rcsb_sample_sequence_length": 350,
                        },
                        "uniprots": [{"rcsb_id": "P1", "gpcrdb_entry_name_slug": "rec_human"}],
                    },
                    {
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["P"]},
                        "rcsb_polymer_entity": {"pdbx_description": "Peptide agonist"},
                        "entity_poly": {
                            "rcsb_entity_polymer_type": "Protein",
                            "rcsb_sample_sequence_length": 34,
                        },
                        "uniprots": [{"rcsb_id": "Q2"}],
                    },
                ]
            }
        }
    }
    tm_by_chain = {"R": {"status": "COMPLETE", "resolved_tms": 7, "total_tms": 7}}
    simplified = prompt_builder.enhanced_simplify_pdb_json(enriched_data, tm_by_chain=tm_by_chain)
    comps = {c["chain_ids"][0]: c for c in simplified["polymer_components"]}

    assert comps["R"]["7tm_status"] == {"R": "COMPLETE 7/7"}
    assert comps["R"]["residue_length"] == 350
    # Peptide: absent from the TM map -> UNKNOWN, and visibly short.
    assert comps["P"]["7tm_status"] == {"P": "UNKNOWN 0/0"}
    assert comps["P"]["residue_length"] == 34


def test_enhanced_simplify_columns_present_without_tm_data():
    # Both columns are ALWAYS emitted; without TM data the status is UNKNOWN 0/0,
    # never invented, and the length still comes from the sequence.
    enriched_data = {
        "data": {
            "entry": {
                "polymer_entities": [
                    {
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["R"]},
                        "rcsb_polymer_entity": {"pdbx_description": "Receptor"},
                        "entity_poly": {
                            "rcsb_entity_polymer_type": "Protein",
                            "rcsb_sample_sequence_length": 350,
                        },
                        "uniprots": [{"rcsb_id": "P1", "gpcrdb_entry_name_slug": "rec_human"}],
                    }
                ]
            }
        }
    }
    simplified = prompt_builder.enhanced_simplify_pdb_json(enriched_data)
    comp = simplified["polymer_components"][0]
    assert comp["7tm_status"] == {"R": "UNKNOWN 0/0"}
    assert comp["residue_length"] == 350


def _receptor_plus_peptide_enriched() -> dict:
    # Chain R is a genuine 7TM receptor (350 aa); chain P is a short peptide
    # agonist (34 aa). Byte-identically shaped, the two differ only in their
    # 7TM facts -- exactly what the fetch must distinguish.
    return {
        "data": {
            "entry": {
                "rcsb_id": "9JR3",
                "polymer_entities": [
                    {
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["R"]},
                        "rcsb_polymer_entity": {"pdbx_description": "Receptor"},
                        "entity_poly": {
                            "rcsb_entity_polymer_type": "Protein",
                            "rcsb_sample_sequence_length": 350,
                        },
                        "uniprots": [{"rcsb_id": "P1", "gpcrdb_entry_name_slug": "rec_human"}],
                    },
                    {
                        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["P"]},
                        "rcsb_polymer_entity": {"pdbx_description": "Peptide agonist"},
                        "entity_poly": {
                            "rcsb_entity_polymer_type": "Protein",
                            "rcsb_sample_sequence_length": 34,
                        },
                        "uniprots": [{"rcsb_id": "Q2"}],
                    },
                ],
            }
        }
    }


def test_enhanced_simplify_omits_7tm_column_when_fetch_failed():
    # FETCH FAILED ENTIRELY: the 7TM column is omitted for EVERY chain so a real
    # receptor is never asserted as a false "UNKNOWN 0/0" (which would be
    # byte-identical to a 34-aa peptide). residue_length is fetch-independent and
    # stays, so the receptor is still visibly the long chain.
    enriched_data = _receptor_plus_peptide_enriched()
    simplified = prompt_builder.enhanced_simplify_pdb_json(
        enriched_data, tm_by_chain=prompt_builder.TM_FETCH_FAILED
    )
    comps = {c["chain_ids"][0]: c for c in simplified["polymer_components"]}
    assert "7tm_status" not in comps["R"]
    assert "7tm_status" not in comps["P"]
    assert comps["R"]["residue_length"] == 350
    assert comps["P"]["residue_length"] == 34


def test_build_prompt_parts_fetch_failure_does_not_make_receptor_look_like_peptide():
    # End-to-end at prompt-build time: the whole-PDB TM fetch fails. The real
    # receptor must NOT render "UNKNOWN 0/0" (the confidently-wrong trap). The
    # column is omitted instead, so the receptor is never made to look like the
    # peptide.
    from unittest.mock import patch

    enriched_data = _receptor_plus_peptide_enriched()
    with patch(
        "gpcr_tools.annotator.prompt_builder.scan_all_chains_7tm",
        return_value=({}, None),  # fetch failed entirely
    ):
        parts = prompt_builder.build_prompt_parts("9JR3", enriched_data, "TEMPLATE")
    joined = "".join(parts)
    assert "UNKNOWN 0/0" not in joined
    assert "7tm_status" not in joined


def test_build_prompt_parts_fetch_success_distinguishes_receptor_from_peptide():
    # When the fetch SUCCEEDS, the column is present: the receptor reads
    # COMPLETE 7/7 and the peptide legitimately reads UNKNOWN 0/0 (a real fact).
    from unittest.mock import patch

    enriched_data = _receptor_plus_peptide_enriched()
    tm_results = {"R": {"status": "COMPLETE", "resolved_tms": 7, "total_tms": 7}}
    with patch(
        "gpcr_tools.annotator.prompt_builder.scan_all_chains_7tm",
        return_value=(tm_results, {"polymer_entities": []}),
    ):
        parts = prompt_builder.build_prompt_parts("9JR3", enriched_data, "TEMPLATE")
    joined = "".join(parts)
    assert '"R": "COMPLETE 7/7"' in joined
    assert '"P": "UNKNOWN 0/0"' in joined


def _enriched_with_assemblies(assemblies):
    return {"data": {"entry": {"rcsb_id": "TEST", "assemblies": assemblies}}}


def test_author_assembly_reference_single_author_defined():
    enriched = _enriched_with_assemblies(
        [
            {
                "rcsb_assembly_container_identifiers": {"assembly_id": "1"},
                "pdbx_struct_assembly": {
                    "rcsb_details": "author_defined_assembly",
                    "method_details": None,
                    "rcsb_candidate_assembly": "Y",
                },
                "rcsb_struct_symmetry": [
                    {
                        "kind": "Global Symmetry",
                        "oligomeric_state": "Hetero 5-mer",
                        "stoichiometry": ["A1", "B1", "C1", "D1", "E1"],
                    }
                ],
            }
        ]
    )
    ref = prompt_builder.generate_author_assembly_reference("TEST", enriched)
    assert "AUTHOR-DEPOSITED BIOLOGICAL ASSEMBLY" in ref
    assert "reference only, NOT authoritative" in ref.replace("not", "NOT")
    assert "Hetero 5-mer" in ref
    assert "author-defined" in ref
    # Stoichiometry renders as a clean [A1, B1, ...] list, not a Python list repr.
    assert "[A1, B1, C1, D1, E1]" in ref
    assert "['A1'" not in ref
    # The reference block warns that a software-predicted / homo-N-mer assembly may
    # reflect crystallographic packing rather than a biological oligomer, and that
    # the model should defer to the paper.
    assert "crystallographic packing" in ref
    assert "defer to the paper" in ref


def test_author_assembly_reference_lists_conflicting_assemblies():
    # An author-defined monomer alongside a software (PISA) homo-dimer: BOTH must
    # be listed, each labeled, so the model sees the conflict and judges itself.
    enriched = _enriched_with_assemblies(
        [
            {
                "rcsb_assembly_container_identifiers": {"assembly_id": "1"},
                "pdbx_struct_assembly": {
                    "rcsb_details": "author_defined_assembly",
                    "method_details": None,
                },
                "rcsb_struct_symmetry": [
                    {
                        "kind": "Global Symmetry",
                        "oligomeric_state": "Monomer",
                        "stoichiometry": ["A1"],
                    }
                ],
            },
            {
                "rcsb_assembly_container_identifiers": {"assembly_id": "2"},
                "pdbx_struct_assembly": {
                    "rcsb_details": "software_defined_assembly",
                    "method_details": "PISA",
                },
                "rcsb_struct_symmetry": [
                    {
                        "kind": "Global Symmetry",
                        "oligomeric_state": "Homo 2-mer",
                        "stoichiometry": ["A2"],
                    }
                ],
            },
        ]
    )
    ref = prompt_builder.generate_author_assembly_reference("TEST", enriched)
    assert "Monomer" in ref
    assert "Homo 2-mer" in ref
    assert "author-defined" in ref
    assert "software-defined (PISA)" in ref


def test_author_assembly_reference_empty_when_no_symmetry():
    # No symmetry block -> nothing to show, so the prompt does not grow.
    enriched = _enriched_with_assemblies(
        [
            {
                "rcsb_assembly_container_identifiers": {"assembly_id": "1"},
                "pdbx_struct_assembly": {"rcsb_details": "author_defined_assembly"},
                "rcsb_struct_symmetry": [],
            }
        ]
    )
    assert prompt_builder.generate_author_assembly_reference("TEST", enriched) == ""
    # Absent assemblies key entirely -> also empty.
    assert prompt_builder.generate_author_assembly_reference("TEST", {}) == ""
