#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 5 折交叉验证结果，计算平均值和标准差。"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np

EXP_DESC = "P:RP:RN:LN=1:1:1:1"
TOTAL_FOLDS = 5

parser = argparse.ArgumentParser(description="汇总 5 折交叉验证结果")
parser.add_argument(
    "--run_id",
    type=str,
    default="",
    help="并行实验专用：用于替代 .current_run_timestamp.txt 的 run_id",
)
args = parser.parse_args()
run_id = (args.run_id or "").strip()

_proj_root = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(_proj_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(
    log_dir, f"aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def aggregate_results():
    logger.info(f"汇总 5 折结果 | {EXP_DESC}")

    timestamp_file = os.path.join(_proj_root, ".current_run_timestamp.txt")
    results_base = os.path.join(_proj_root, "results")

    if run_id:
        results_dir = f"{results_base}_{run_id}"
    elif os.path.isfile(timestamp_file):
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamp = f.read().strip()
        results_dir = f"{results_base}_{timestamp}"
    else:
        results_dir = results_base
        logger.warning(f"未找到时间戳文件，使用默认目录: {results_dir}")

    os.makedirs(results_dir, exist_ok=True)

    metric_names = ["AUC", "AUPR", "F1", "Recall", "Precision", "Accuracy"]
    metrics_list = {name: [] for name in metric_names}
    loaded_folds = 0

    for fold in range(TOTAL_FOLDS):
        results_file = os.path.join(results_dir, f"fold_{fold}_results.json")
        if not os.path.isfile(results_file):
            logger.warning(f"未找到第 {fold} 折: {results_file}")
            continue
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                fold_result = json.load(f)
            for name in metric_names:
                val = fold_result["metrics"].get(name)
                if val is not None:
                    metrics_list[name].append(val)
            loaded_folds += 1
            logger.info(f"  第 {fold} 折已加载")
        except Exception as exc:
            logger.warning(f"  第 {fold} 折加载失败: {exc}")

    if loaded_folds == 0:
        logger.error("未找到任何结果文件！")
        return None

    logger.info(f"成功加载 {loaded_folds} 折")

    final_metrics = {}
    logger.info("=" * 60)
    for name, values in metrics_list.items():
        if values:
            mean_val, std_val = np.mean(values), np.std(values)
            final_metrics[name] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "values": [float(v) for v in values],
                "count": len(values),
            }
            logger.info(
                f"{name:12s}: {mean_val:.6f} +/- {std_val:.6f} (n={len(values)})"
            )
        else:
            final_metrics[name] = {
                "mean": None,
                "std": None,
                "values": [],
                "count": 0,
            }
            logger.warning(f"{name:12s}: 无有效数据")
    logger.info("=" * 60)

    final_results = {
        "experiment_description": EXP_DESC,
        "completed_folds": loaded_folds,
        "total_folds": TOTAL_FOLDS,
        "metrics": final_metrics,
    }

    final_json = os.path.join(results_dir, "final_results.json")
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    final_txt = os.path.join(results_dir, "final_results.txt")
    with open(final_txt, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{EXP_DESC}\n")
        f.write(
            f"完成: {loaded_folds}/{TOTAL_FOLDS} 折 | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"{'=' * 60}\n\n")
        f.write("评估指标 (mean +/- std):\n")
        f.write("-" * 60 + "\n")
        for name, data in final_metrics.items():
            if data["mean"] is not None:
                f.write(
                    f"{name:12s}: {data['mean']:.6f} +/- {data['std']:.6f} "
                    f"(n={data['count']})\n"
                )
            else:
                f.write(f"{name:12s}: 无有效数据\n")
        f.write(f"\n各折详细结果:\n{'-' * 60}\n")
        for name in metric_names:
            if final_metrics[name]["values"]:
                f.write(f"\n{name}:\n")
                for i, val in enumerate(final_metrics[name]["values"]):
                    f.write(f"  Fold {i}: {val:.6f}\n")

    logger.info(f"结果已保存: {final_json}")
    logger.info(f"文本已保存: {final_txt}")

    print(f"\n{'=' * 60}")
    print(f"{EXP_DESC} | {loaded_folds}/{TOTAL_FOLDS} 折")
    print("=" * 60)
    for name, data in final_metrics.items():
        if data["mean"] is not None:
            print(f"{name:12s}: {data['mean']:.6f} +/- {data['std']:.6f}")
    print("=" * 60)

    return final_results


if __name__ == "__main__":
    aggregate_results()
