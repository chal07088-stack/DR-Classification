import argparse
import json

from drpair.runtime import load_config


def main():
    parser = argparse.ArgumentParser(description="DR paired-organ PHASE 1 workflow")
    parser.add_argument("--config", default="configs/phase1.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Validate official data and write patient manifests")
    commands.add_parser("status", help="Show existing checkpoints without training")
    experts = commands.add_parser("experts", help="Explicitly start expert learning")
    selection = experts.add_mutually_exclusive_group(required=True)
    selection.add_argument("--job", help="final_single, final_paired, oof1_single, ...")
    selection.add_argument("--all", action="store_true")
    commands.add_parser("aggregate-oof", help="Aggregate completed patient OOF prediction files")
    commands.add_parser(
        "gates", help="Teacher A/B and validation cross-fitting on saved predictions"
    )
    commands.add_parser("report", help="Evaluate all fixed arms and end PHASE 1")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        from pathlib import Path

        from drpair.data.manifest import prepare

        result = prepare(
            cfg["data"]["root"],
            cfg["data"]["manifest"],
            cfg["data"]["fold_manifest"],
            Path(cfg["data"]["manifest"]).parent / "audit.json",
        )
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        from drpair.experiments.experts import status

        print(json.dumps(status(cfg), indent=2))
    elif args.command == "experts":
        from drpair.experiments.experts import expert_jobs, run_experts

        run_experts(cfg, expert_jobs() if args.all else [args.job])
    elif args.command == "aggregate-oof":
        from drpair.experiments.experts import aggregate_oof

        aggregate_oof(cfg)
    elif args.command == "gates":
        from drpair.experiments.gate_study import run_gate_study

        run_gate_study(cfg)
    elif args.command == "report":
        from drpair.evaluation.report import make_report

        print(make_report(cfg))


if __name__ == "__main__":
    main()
