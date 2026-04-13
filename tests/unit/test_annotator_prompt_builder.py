from gpcr_tools.annotator import prompt_builder


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
