import logging
from pathlib import Path

from gpcr_tools.annotator.runner import build_and_submit_batch, run_single_pdb
from gpcr_tools.config import get_config
from gpcr_tools.fetcher.targets import read_targets

logger = logging.getLogger(__name__)


def run_annotate(
    pdb_id: str | None,
    targets_file: str | None,
    prompt_file: str | None,
    num_runs: int,
    use_batch: bool,
) -> None:
    """CLI handler for 'gpcr-tools annotate'."""
    config = get_config()

    # Resolve prompt
    if prompt_file:
        prompt_path = Path(prompt_file).resolve()
    else:
        prompt_path = config.default_prompt_file

    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        return

    with open(prompt_path) as f:
        prompt_text = f.read().strip()

    # Resolve targets
    if pdb_id:
        targets = [pdb_id.upper()]
    elif targets_file:
        targets = read_targets(Path(targets_file))
    else:
        # auto-discover from enriched missing complete ai_results
        targets = []
        if config.enriched_dir.exists():
            for p in config.enriched_dir.glob("*.json"):
                target_pdb = p.stem
                out_dir = config.ai_results_dir / target_pdb

                # Check runs
                completed = 0
                if out_dir.exists():
                    for n in range(1, num_runs + 1):
                        if (out_dir / f"run_{n}.json").exists():
                            completed += 1

                if completed < num_runs:
                    targets.append(target_pdb)

    if not targets:
        logger.info("No targets to annotate. All done!")
        return

    if use_batch:
        build_and_submit_batch(targets, prompt_text, num_runs)
    else:
        import json

        for target_id in targets:
            enriched_file = config.enriched_dir / f"{target_id}.json"
            pdf_file = config.papers_dir / f"{target_id}.pdf"

            if not enriched_file.exists():
                logger.warning(f"[{target_id}] enriched data not found, skipping")
                continue

            if not pdf_file.exists():
                logger.warning(f"[{target_id}] PDF not found, skipping")
                continue

            with open(enriched_file) as f:
                enriched_data = json.load(f)

            logger.info(f"[{target_id}] Starting annotation ({num_runs} runs)...")
            run_single_pdb(target_id, enriched_data, prompt_text, pdf_file, num_runs)
