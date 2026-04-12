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
                                            "description": "The state value. Must be one of the specified enum values.",
                                            "enum": [
                                                "inactive",
                                                "active",
                                                "other",
                                                "intermediate",
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
                            },
                            "required": ["uniprot_entry_name", "chain_id"],
                        },
                        "ligands": {
                            "type": "array",
                            "description": "A list of ALL ligands. Must include an 'Apo' entry if no ligand is present.",
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
                                                "description": "The role value. Must be one of the specified enum values.",
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
                                },
                                "required": [
                                    "name",
                                    "chem_comp_id",
                                    "chain_id",
                                    "role",
                                    "type",
                                    "pubchem_id",
                                    "synonyms",
                                ],
                            },
                        },
                        "signaling_partners": {
                            "type": "object",
                            "description": "Container for all signaling partners. Can be empty.",
                            "properties": {
                                "g_protein": {
                                    "type": "object",
                                    "description": "G-protein heterotrimer details. Omit if not present.",
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
                                            "description": "Any notes, e.g., 'Engineered G protein', 'Gs/Gi chimera'. If is_chimera is true, briefly explain the chimera's composition here.",
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
                            "description": "List of other non-GPCR, non-signaling proteins. Can be an empty list.",
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

TOOL_CONFIG = types.GenerateContentConfig(
    tools=[ANNOTATION_TOOL],
    temperature=0.0,
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="ANY")  # type: ignore[arg-type]
    ),
)
