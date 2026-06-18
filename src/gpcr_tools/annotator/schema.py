"""Gemini tool schema definition for GPCR structure annotation."""

from __future__ import annotations

from google.genai import types

ANNOTATION_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            **{  # type: ignore[arg-type]
                "name": "annotate_gpcr_db_structure",
                "description": "Extracts and structures all key information for a GPCR structure from a scientific paper and PDB metadata, preparing it for direct import into the GPCRdb. For any field requiring inference, confidence and evidence must be provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_info": {
                            "type": "object",
                            "description": "Core details about the PDB structure itself, primarily from PDB metadata.",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "description": "The experimental method used. Must be one of the specified values.",
                                    "enum": [
                                        "Electron crystallography",
                                        "Electron microscopy",
                                        "Refined CEM",
                                        "Refined X-ray",
                                        "X-ray diffraction",
                                    ],
                                },
                                "resolution": {
                                    "type": "number",
                                    "description": "Resolution in Angstroms.",
                                },
                                "release_date": {
                                    "type": "string",
                                    "description": "Initial release date (YYYY-MM-DD).",
                                },
                                "state": {
                                    "type": "object",
                                    "description": "Inferred functional state of the receptor, with confidence and evidence.",
                                    "properties": {
                                        "value": {
                                            "type": "string",
                                            "description": "The state value. Must be one of the specified enum values. Use 'unknown' if neither the paper nor the metadata establishes the functional state — do not guess.",
                                            "enum": [
                                                "inactive",
                                                "active",
                                                "other",
                                                "intermediate",
                                                "unknown",
                                            ],
                                        },
                                        "confidence": {
                                            "type": "string",
                                            "description": "Confidence level of this inference.",
                                            "enum": ["High", "Medium", "Low"],
                                        },
                                        "evidence": {
                                            "type": "object",
                                            "description": "The justification for the state assignment.",
                                            "properties": {
                                                "source": {
                                                    "type": "string",
                                                    "description": "The source of the evidence.",
                                                    "enum": [
                                                        "Paper",
                                                        "PDB Metadata",
                                                        "Both Paper and PDB Metadata",
                                                    ],
                                                },
                                                "quote_or_path": {
                                                    "type": "string",
                                                    "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                },
                                                "reasoning": {
                                                    "type": "string",
                                                    "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                },
                                            },
                                            "required": [
                                                "source",
                                                "quote_or_path",
                                                "reasoning",
                                            ],
                                        },
                                    },
                                    "required": ["value", "confidence", "evidence"],
                                },
                                "note": {
                                    "type": "string",
                                    "description": "Any specific notes, e.g., 'Fragment', 'Engineered'.",
                                },
                            },
                            "required": ["method", "resolution", "release_date", "state"],
                        },
                        "receptor_info": {
                            "type": "object",
                            "description": "Information about the main GPCR receptor protein.",
                            "properties": {
                                "uniprot_entry_name": {
                                    "type": "string",
                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                },
                                "chain_id": {
                                    "type": "string",
                                    "description": "The chain ID of the GPCR.",
                                },
                                "oligomeric_state": {
                                    "type": "object",
                                    "description": "Inferred oligomeric state of the GPCR RECEPTOR(S) themselves, with confidence and evidence.",
                                    "properties": {
                                        "value": {
                                            "type": "string",
                                            "description": (
                                                "How many GPCR receptor copies form the biological unit, and whether they are the same or different receptors. "
                                                "Count ONLY the GPCR receptor(s); do NOT count G-protein, arrestin, nanobody, antibody, peptide, or ligand partners — "
                                                "they are not receptor protomers. Use the per-chain 7TM status and residue length in the polymer table to tell a true "
                                                "7TM receptor from a non-receptor partner, and weigh the paper and the (reference-only) author biological assembly.\n"
                                                "'monomer' = one receptor copy (a single receptor with a G-protein heterotrimer is still 'monomer' — the Gα/β/γ are partners, "
                                                "not receptor copies, even when the author assembly is reported as a 'Hetero 5-mer').\n"
                                                "'homo-dimer' / 'homo-trimer' / 'homo-tetramer' = two / three / four copies of the SAME receptor (e.g. a mGlu2 or CaSR receptor dimer is 'homo-dimer').\n"
                                                "'hetero-dimer' = two DIFFERENT receptor subunits forming one obligate receptor (e.g. the GABA-B receptor, GBR1 + GBR2).\n"
                                                "'unknown' = the receptor oligomeric state cannot be established from the paper or metadata — do not guess.\n"
                                                "Distinguish a genuine biological oligomer from incidental crystallographic copies of a monomer; when the evidence does not settle it, prefer 'unknown' over a guess."
                                            ),
                                            "enum": [
                                                "monomer",
                                                "homo-dimer",
                                                "homo-trimer",
                                                "homo-tetramer",
                                                "hetero-dimer",
                                                "unknown",
                                            ],
                                        },
                                        "confidence": {
                                            "type": "string",
                                            "description": "Confidence level of this inference.",
                                            "enum": ["High", "Medium", "Low"],
                                        },
                                        "evidence": {
                                            "type": "object",
                                            "description": "The justification for the oligomeric-state assignment.",
                                            "properties": {
                                                "source": {
                                                    "type": "string",
                                                    "description": "The source of the evidence.",
                                                    "enum": [
                                                        "Paper",
                                                        "PDB Metadata",
                                                        "Both Paper and PDB Metadata",
                                                    ],
                                                },
                                                "quote_or_path": {
                                                    "type": "string",
                                                    "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                },
                                                "reasoning": {
                                                    "type": "string",
                                                    "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                },
                                            },
                                            "required": [
                                                "source",
                                                "quote_or_path",
                                                "reasoning",
                                            ],
                                        },
                                    },
                                    "required": ["value", "confidence", "evidence"],
                                },
                            },
                            "required": ["uniprot_entry_name", "chain_id", "oligomeric_state"],
                        },
                        "ligands": {
                            "type": "array",
                            "description": "A list of ALL ligands. Must include an 'Apo' entry if no ligand is present. A ligand is any entity — small molecule, peptide, or protein — that binds the receptor and acts on its function (agonist / antagonist / PAM / NAM / allosteric modulator); this includes a functional protein or peptide binder (e.g. a protein agonist such as R-spondin, or an activating antibody/nanobody). A G-protein-derived or transducer-mimetic peptide is not a ligand.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Common name of the ligand. Use 'Apo' for empty structures.",
                                    },
                                    "chain_id": {
                                        "type": "string",
                                        "description": "Chain ID, if applicable. Can be 'None'.",
                                    },
                                    "role": {
                                        "type": "object",
                                        "description": "Inferred pharmacological role of the ligand, with confidence and evidence.",
                                        "properties": {
                                            "value": {
                                                "type": "string",
                                                "description": (
                                                    "The ligand's pharmacological role. Must be one of the specified enum values. Decide by site and mechanism, not by wording habit.\n"
                                                    "Orthosteric site: activates → 'Agonist' (partial → 'Agonist (partial)';\n"
                                                    "requires a second obligate agonist → 'Co-agonist');\n"
                                                    "blocks the agonist → 'Antagonist';\n"
                                                    "lowers constitutive activity → 'Inverse agonist'.\n"
                                                    "Non-orthosteric site (allosteric / intracellular / outer surface):\n"
                                                    "activates from there → 'Allosteric agonist';\n"
                                                    "activates and also potentiates the orthosteric agonist (the paper shows both) → 'Ago-PAM';\n"
                                                    "only potentiates or only inhibits the orthosteric response with no intrinsic activation → 'PAM' / 'NAM';\n"
                                                    "blocks from an allosteric site → 'Allosteric antagonist'.\n"
                                                    "Default to 'Allosteric agonist' over 'Ago-PAM' unless the paper explicitly shows both standalone activation and potentiation.\n"
                                                    "Not a pharmacological ligand (a structural lipid, detergent, or cofactor) → 'Cofactor';\n"
                                                    "truly undetermined → 'unknown'.\n"
                                                    "An endogenous peptide or protein hormone binding the extracellular domain (e.g. FSH) is an 'Agonist', not 'Allosteric agonist'."
                                                ),
                                                "enum": [
                                                    "Antagonist",
                                                    "PAM",
                                                    "NAM",
                                                    "Ago-PAM",
                                                    "Allosteric antagonist",
                                                    "Apo (no ligand)",
                                                    "Cofactor",
                                                    "unknown",
                                                    "Inverse agonist",
                                                    "Agonist",
                                                    "Allosteric agonist",
                                                    "Co-agonist",
                                                    "Agonist (partial)",
                                                ],
                                            },
                                            "confidence": {
                                                "type": "string",
                                                "description": "Confidence level of this inference.",
                                                "enum": ["High", "Medium", "Low"],
                                            },
                                            "evidence": {
                                                "type": "object",
                                                "description": "The justification for the role assignment.",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "The source of the evidence.",
                                                        "enum": [
                                                            "Paper",
                                                            "PDB Metadata",
                                                            "Both Paper and PDB Metadata",
                                                        ],
                                                    },
                                                    "quote_or_path": {
                                                        "type": "string",
                                                        "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                    },
                                                    "reasoning": {
                                                        "type": "string",
                                                        "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                    },
                                                },
                                                "required": [
                                                    "source",
                                                    "quote_or_path",
                                                    "reasoning",
                                                ],
                                            },
                                        },
                                        "required": ["value", "confidence", "evidence"],
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Molecular type. prioritize info from 'gpcrdb_determined_type', but must be one of the specified values.",
                                        "enum": [
                                            "lipid",
                                            "na",
                                            "none",
                                            "peptide",
                                            "protein",
                                            "small-molecule",
                                        ],
                                    },
                                    "pubchem_id": {
                                        "type": "string",
                                        "description": "'gpcrdb_pubchem_cid' if available, otherwise try to find the correct PubChem ID in the paper. Use 'None' if not applicable.",
                                    },
                                    "chem_comp_id": {
                                        "type": "string",
                                        "description": "The chemical component ID (e.g., 'CAU', 'CLR') from the PDB metadata. Use 'None' for Apo entries.",
                                    },
                                    "synonyms": {
                                        "type": "array",
                                        "description": "List of synonyms from the PDB metadata. Can be an empty list.",
                                        "items": {"type": "string"},
                                    },
                                    "site_ref": {
                                        "type": "string",
                                        "description": (
                                            "The binding site of this molecule. site_ref records WHERE this molecule sits — a pure POSITION, never what it does. Any role (agonist, modulator, or other) can occur at any position, pharmacology is recorded only in 'role'. Infer the position from the DETECTOR EVIDENCE geometry plus the paper. If neither settles it, mark 'unknown'. If one molecule is modelled at more than one distinct site, emit a separate entry per site.\n"
                                            "\n"
                                            "Boundaries and Tie-breaks of each value (all are positions):\n"
                                            "* 'orthosteric' = the pocket where the receptor's OWN endogenous agonist binds. For most Class A (aminergic, lipid, nucleotide, opsin, most peptide) and for the adhesion (Class B2) tethered-stalk peptide, this is the 7TM core pocket, so a molecule there is 'orthosteric'. BUT where the endogenous agonist instead binds an extracellular domain — glycoprotein-hormone receptors (FSHR/LHR/TSHR, LRR ectodomain) and Class C (Venus-flytrap) — the 7TM pocket is NOT orthosteric: a small molecule there is 'allosteric_7tm', and the endogenous-agonist ECD/VFT site is 'extracellular_domain' (record its agonism in 'role').\n"
                                            "* 'extracellular_vestibule' = the vestibule just above the orthosteric pocket. Tie-break: if the molecule contacts the orthosteric core-anchor residues, choose 'orthosteric'.\n"
                                            "* 'allosteric_7tm' = a discrete non-orthosteric pocket or groove on the 7TM bundle. Use the 'enclosure' and 'pocket-vs-lipid-facing' evidence to separate a defined groove (this value) from a flat exposed surface ('membrane_facing').\n"
                                            "* 'intracellular' = on the receptor cytoplasmic face / transducer interface (around DRY-Arg 3.50, helix 8, intracellular TM ends). Tie-break: if the molecule sits on the cytoplasmic side, choose 'intracellular' over 'allosteric_7tm' even when its pharmacological action could be allosteric.\n"
                                            "* 'extracellular_domain' = the extracellular domain (ECD), Venus-flytrap (VFT), or N-terminal domain, outside the 7TM bundle.\n"
                                            "* 'membrane_facing' = the lipid-exposed outer wall of the 7TM bundle (the bulk-bilayer surface), where a molecule lies flat/exposed rather than in a defined pocket."
                                        ),
                                        "enum": [
                                            "orthosteric",
                                            "allosteric_7tm",
                                            "extracellular_vestibule",
                                            "intracellular",
                                            "extracellular_domain",
                                            "membrane_facing",
                                            "unknown",
                                        ],
                                    },
                                    "site_ref_justification": {
                                        "type": "string",
                                        "description": "Brief free-text justification for the site_ref call: which geometric facts and/or paper statements support it. Optional; omit if the site is obvious or unknown.",
                                    },
                                },
                                "required": [
                                    "name",
                                    "chem_comp_id",
                                    "chain_id",
                                    "role",
                                    "type",
                                    "pubchem_id",
                                    "synonyms",
                                    "site_ref",
                                ],
                            },
                        },
                        "signaling_partners": {
                            "type": "object",
                            "description": "Container for all signaling partners. Can be empty.",
                            "properties": {
                                "g_protein": {
                                    "type": "object",
                                    "description": "G-protein heterotrimer details. Omit if not present. When only a fragment of the Gα subunit is present — its C-terminal / α5 helix, or a transducer-mimetic peptide that substitutes for the full subunit (e.g. a GαCT peptide) — it still belongs here as the alpha_subunit; record the fragment's UniProt entry name and chain ID in alpha_subunit, and note in the g_protein-level 'note' field that only the C-terminal/α5 fragment is modelled. Do not place such a peptide in ligands or auxiliary_proteins.",
                                    "properties": {
                                        "alpha_subunit": {
                                            "type": "object",
                                            "properties": {
                                                "uniprot_entry_name": {
                                                    "type": "string",
                                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                                },
                                                "chain_id": {"type": "string"},
                                            },
                                            "required": ["uniprot_entry_name", "chain_id"],
                                        },
                                        "beta_subunit": {
                                            "type": "object",
                                            "properties": {
                                                "uniprot_entry_name": {
                                                    "type": "string",
                                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                                },
                                                "chain_id": {"type": "string"},
                                            },
                                            "required": [
                                                "uniprot_entry_name",
                                                "chain_id",
                                            ],
                                        },
                                        "gamma_subunit": {
                                            "type": "object",
                                            "properties": {
                                                "uniprot_entry_name": {
                                                    "type": "string",
                                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                                },
                                                "chain_id": {"type": "string"},
                                            },
                                            "required": [
                                                "uniprot_entry_name",
                                                "chain_id",
                                            ],
                                        },
                                        "is_chimeric": {
                                            "type": "boolean",
                                            "description": "Set to true if the PDB metadata or paper indicates this is an engineered chimeric G protein.",
                                        },
                                        "note": {
                                            "type": "string",
                                            "description": "Any notes, e.g., 'Engineered G protein', 'Gs/Gi chimera'. If is_chimeric is true, briefly explain the chimera's composition here. SOURCING REQUIREMENT: the specific composition details MUST be stated in the paper or PDB metadata.",
                                        },
                                    },
                                    "required": ["alpha_subunit"],
                                },
                                "arrestin": {
                                    "type": "object",
                                    "description": "Arrestin details. Omit if not present.",
                                    "properties": {
                                        "uniprot_entry_name": {
                                            "type": "string",
                                            "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                        },
                                        "chain_id": {"type": "string"},
                                        "note": {"type": "string"},
                                    },
                                    "required": ["uniprot_entry_name", "chain_id"],
                                },
                            },
                        },
                        "auxiliary_proteins": {
                            "type": "array",
                            "description": "List of proteins present for structural, stabilization, or expression reasons that do not themselves modulate receptor activity — crystallization fusions (BRIL / cytochrome b562, T4 lysozyme), stabilizing Fab / nanobody / scFv, and other non-functional, non-signaling proteins. Can be an empty list. Deciding test: if the protein binds the receptor to modulate its activity, put it in ligands; if it is only a structural/stabilization/expression aid, put it here. A stabilizing nanobody is auxiliary; an activating nanobody is a ligand. Read the paper; absent a functional claim, default to auxiliary. (Signaling proteins like G protein and arrestin belong in signaling_partners, not here — including a Gα-derived C-terminal/α5 or transducer-mimetic peptide, which goes in signaling_partners.g_protein.alpha_subunit.)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The specific, standardized name of the protein (e.g., 'T4-Lysozyme', 'Nanobody-35').",
                                    },
                                    "type": {
                                        "type": "object",
                                        "description": "The functional category of the protein, with confidence and evidence.",
                                        "properties": {
                                            "value": {
                                                "type": "string",
                                                "description": "The category value. Must be one of the specified enum values.",
                                                "enum": [
                                                    "Fusion protein",
                                                    "Nanobody",
                                                    "Antibody",
                                                    "scFv",
                                                    "Antibody fab fragment",
                                                    "GRK",
                                                    "RAMP",
                                                    "MRAP",
                                                    "DARPin",
                                                    "Other",
                                                ],
                                            },
                                            "confidence": {
                                                "type": "string",
                                                "description": "Confidence level of this classification.",
                                                "enum": ["High", "Medium", "Low"],
                                            },
                                            "evidence": {
                                                "type": "object",
                                                "description": "The justification for the type assignment.",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "The source of the evidence.",
                                                        "enum": [
                                                            "Paper",
                                                            "PDB Metadata",
                                                            "Both Paper and PDB Metadata",
                                                        ],
                                                    },
                                                    "quote_or_path": {
                                                        "type": "string",
                                                        "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                    },
                                                    "reasoning": {
                                                        "type": "string",
                                                        "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                    },
                                                },
                                                "required": [
                                                    "source",
                                                    "quote_or_path",
                                                    "reasoning",
                                                ],
                                            },
                                        },
                                        "required": ["value", "confidence", "evidence"],
                                    },
                                    "chain_id": {
                                        "type": "string",
                                        "description": "Chain ID(s). Can be comma-separated if multiple chains.",
                                    },
                                },
                                "required": ["name", "type", "chain_id"],
                            },
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of 2-3 novel and specific scientific insights from the paper's structural analysis.",
                        },
                    },
                    "required": [
                        "structure_info",
                        "receptor_info",
                        "ligands",
                        "key_findings",
                    ],
                },
            }
        )
    ]
)

# Optional per-ligand field injected by detect_orchestrator.build_tool_for_signals
# ONLY when an incidental-candidate signal is present (e.g. cholesterol, palmitate).
# It is never part of the base tool, so an ordinary structure's schema is unchanged.
PHARMACOLOGICAL_ROLE_CHECK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    description=(
        "Only for a detector-flagged incidental-candidate molecule (e.g. cholesterol, "
        "palmitate): your judgment of whether it is a functional ligand or an "
        "incidental / structural component, with the evidence."
    ),
    properties={
        "is_functional_ligand": types.Schema(
            type=types.Type.BOOLEAN,
            description="True if it acts as a functional ligand; False if incidental / structural.",
        ),
        "confidence": types.Schema(type=types.Type.STRING, enum=["High", "Medium", "Low"]),
        "evidence": types.Schema(
            type=types.Type.STRING,
            description="A direct paper quote or brief reasoning supporting the judgment.",
        ),
    },
)

# No temperature override here: the default model is tuned to run at its own
# default temperature, and pinning a low value can trigger reasoning loops or
# degrade quality on complex inputs. Leaving it unset uses the model default;
# callers may still set one explicitly (annotate --temperature).
TOOL_CONFIG = types.GenerateContentConfig(
    tools=[ANNOTATION_TOOL],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="ANY")  # type: ignore[arg-type]
    ),
)
