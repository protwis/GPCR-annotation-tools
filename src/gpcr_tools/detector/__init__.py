"""Pre-annotation detect stage.

Runs structural / bioinformatic checks on a PDB's enriched metadata before
annotation (no AI, no paper) and emits ``DetectSignal`` records. Signals are
persisted per PDB and later drive prompt branching and human-review routing.
"""
