from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_data import audit_unified
from .build_features import build_feature_caches
from .data_adapters import CATPRED_FILES, adapt_catpred
from .evaluate import aggregate_oof, evaluate_predictions_csv
from .predict import predict_file
from .train import run_experiment
from .utils import project_path, read_yaml


def cmd_adapt(args: argparse.Namespace) -> None:
    tasks = [args.task] if args.task else list(CATPRED_FILES)
    summaries = []
    for task in tasks:
        output = Path(args.output_dir) / f"{task}_unified.csv"
        summary = adapt_catpred(task, args.catpred_root, output, include_splits=not args.no_splits, limit=args.limit)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))


def cmd_audit(args: argparse.Namespace) -> None:
    roots = [project_path(p) for p in args.structure_root] if args.structure_root else None
    root = roots[0] if roots else None
    report = audit_unified(args.input, args.output_dir, structure_root=root)
    print(json.dumps({k: report[k] for k in ["n_rows", "unique_sequence_ids", "unique_chemical_contexts", "unique_reactions", "label_missing_rows", "structure_local_existing_rows"]}, ensure_ascii=False))


def cmd_build_features(args: argparse.Namespace) -> None:
    config = read_yaml(args.config)
    configured_index_dir = config.get("features", {}).get("index_dir")
    if not configured_index_dir:
        data_index = config.get("data", {}).get("feature_index_csv")
        configured_index_dir = str(Path(data_index).parent) if data_index else "features_cache/index"
    output_dir = project_path(args.output_dir) if args.output_dir else project_path(configured_index_dir)
    summary = build_feature_caches(
        project_path(config["data"]["unified_csv"]),
        output_dir,
        config,
        device=args.device,
        limit=args.limit,
        task=args.task or config.get("task", config.get("data", {}).get("task", "kcat")),
    )
    print(json.dumps({k: summary[k] for k in ["n_rows", "status_counts", "failure_counts"]}, ensure_ascii=False))


def cmd_train(args: argparse.Namespace) -> None:
    summary = run_experiment(args.config, output_root=args.output_root, resume=args.resume, limit=args.limit)
    print(json.dumps({k: summary[k] for k in ["experiment", "n_runs", "metrics_mean"]}, ensure_ascii=False))


def cmd_evaluate(args: argparse.Namespace) -> None:
    if args.oof_files:
        report = aggregate_oof(args.oof_files, args.output_dir)
    else:
        report = evaluate_predictions_csv(args.predictions, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, default=str))


def cmd_predict(args: argparse.Namespace) -> None:
    frame = predict_file(
        args.config,
        args.checkpoint,
        args.unified_csv,
        args.feature_index_csv,
        args.output_csv,
        cache_root=args.cache_root,
        task=args.task,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps({"n_predictions": int(len(frame)), "output_csv": str(args.output_csv)}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="酶动力学绝对值预测工程 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("adapt", help="CatPred 数据适配为统一表")
    p.add_argument("--catpred-root", required=True)
    p.add_argument("--task", choices=sorted(CATPRED_FILES), default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-splits", action="store_true")
    p.set_defaults(func=cmd_adapt)

    p = sub.add_parser("audit", help="统一表数据审计")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--structure-root", action="append", default=[])
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("build-features", help="生成并缓存特征")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("train", help="训练/恢复训练")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="评价预测结果")
    p.add_argument("--predictions", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--oof-files", nargs="*", default=[])
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="新样本推理")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unified-csv", required=True)
    p.add_argument("--feature-index-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
