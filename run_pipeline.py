import argparse
from pathlib import Path

from scripts import t01_data_acquisition as t1
from scripts import t02_graph_construction as t2
from scripts import t03_od_matrix as t3
from scripts import t04_2sfca as t4
from scripts import t05_classification_mapping as t5

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


STAGES = ("t1", "t2", "t3", "t4", "t5")


def run_stage(stage: str) -> None:
    """Run one stage after its own input validation."""
    if stage == "t1":
        print("Running T1: data acquisition")
        t1.run(RAW_DATA_DIR, DATA_DIR)
    elif stage == "t2":
        print("Running T2: graph construction")
        t2.run(DATA_DIR)
    elif stage == "t3":
        print("Running T3: OD matrices")
        t3.run(DATA_DIR)
    elif stage == "t4":
        print("Running T4: accessibility scores")
        t4.run(DATA_DIR)
    elif stage == "t5":
        print("Running T5: classification and mapping")
        t5.run(DATA_DIR, OUTPUTS_DIR)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    """Run all stages or a selected stage from the project root."""
    parser = argparse.ArgumentParser(description="Run the tea accessibility pipeline.")
    parser.add_argument("--stage", choices=STAGES, help="Run only one stage.")
    parser.add_argument("--from-stage", choices=STAGES, help="Run from this stage through T5.")
    args = parser.parse_args()

    if args.stage and args.from_stage:
        parser.error("--stage and --from-stage cannot be used together")
    if args.stage:
        stages = (args.stage,)
    elif args.from_stage:
        stages = STAGES[STAGES.index(args.from_stage):]
    else:
        stages = STAGES

    for stage in stages:
        run_stage(stage)
    print("Pipeline complete")


if __name__ == "__main__":
    main()