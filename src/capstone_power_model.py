#!/usr/bin/env python3
"""
Capstone synthetic power model workflow for Figure 5/9 checks and Figure 10.

This script provides two commands:

1. Generate a deterministic synthetic PTPX-like dataset for testing the hierarchical Capstone model:

   python3 capstone_power_model.py make-synthetic \
       --output-dir data/power_model_synthetic \
       --seed 7

2. Fit and evaluate the hierarchical Capstone model:

   python3 capstone_power_model.py evaluate \
       --manifest data/power_model_synthetic/manifest.json \
       --output-dir figures/generated_power_model

Dependencies:

    python3 -m pip install numpy scipy matplotlib

The fitted hierarchical model is

    Y[r,k] ~= gamma[k] * sum_e B[r,e] X[e,k]

with B >= 0 and a structural event-to-row mask.  This is equivalent to the
paper's W/alpha parameterization:

    alpha[e] = sum_r B[r,e]
    W[r,e]   = B[r,e] / alpha[e]

when alpha[e] is nonzero.  Leakage is fit separately with nonnegative ridge
regression.  Deployment replaces fitted per-sample gamma with a compiler-side
activity proxy:

    log(gamma_hat) = a * log(proxy) + b.

Figure 5 compares the all-workload fit with the VecAdd-only
comparison. Figure 9 trains on split=train samples and
evaluates both train and held-out workloads, and compares the deployable model
with an in-sample oracle that uses fitted gamma within the same abstraction.
Figure 10 reports the learned event-to-row mapping and compares the selected
workload's synthetic signoff data and model-predicted category breakdowns.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import lsq_linear


SCHEMA_VERSION = 1
EPS = 1e-12

FEATURE_ORDER = [
    "num_pe_tiles",
    "num_pe_ports",
    "num_mem_tiles",
    "num_mem_ports",
    "num_io_tiles",
    "num_ic_rmux",
    "num_ic_reg",
    "num_ic_port",
    "num_ic_sb",
    "num_pipeline_regs",
    "bias",
]

LEAK_FEATURE_ORDER = [
    "num_pe_tiles",
    "num_mem_tiles",
    "num_io_tiles",
    "num_pipeline_regs",
    "bias",
]

ROW_GROUPS = [
    "PE",
    "MEM",
    "IO",
    "SB",
    "RMUX",
    "REG",
    "PIPE",
    "PORT",
    "OTHER",
    "SELF",
]

FIGURE10_CATEGORIES = [
    "PE",
    "MEM",
    "IO",
    "SB",
    "RMUX",
    "REG",
    "PORT",
    "OTHER",
]

FEATURE_GROUP = {
    "num_pe_tiles": "PE",
    "num_pe_ports": "PE",
    "num_mem_tiles": "MEM",
    "num_mem_ports": "MEM",
    "num_io_tiles": "IO",
    "num_ic_rmux": "RMUX",
    "num_ic_reg": "REG",
    "num_ic_port": "PORT",
    "num_ic_sb": "SB",
    "num_pipeline_regs": "PIPE",
}

DEFAULT_PROXY_WEIGHTS = {
    "num_ic_reg": 1.0,
    "num_pipeline_regs": 1.0,
    "num_io_tiles": 0.1,
    "num_ic_rmux": 0.5,
    "num_ic_sb": 0.5,
    "num_ic_port": 0.25,
    "num_pe_tiles": 0.1,
    "num_mem_tiles": 0.2,
    "num_pe_ports": 0.05,
    "num_mem_ports": 0.05,
}

LINE_RE = re.compile(
    r"""
    ^(?P<indent>\s*)
    (?P<name>.+?)
    \s+
    (?P<internal>[0-9.eE+-]+)\s+
    (?P<switching>[0-9.eE+-]+)\s+
    (?P<leakage>[0-9.eE+-]+)\s+
    (?P<total>[0-9.eE+-]+)\s+
    (?P<percent>[0-9.eE+-]+)
    \s*$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class HierRow:
    name: str
    indent: int
    internal_mw: float
    switching_mw: float
    leakage_mw: float
    total_mw: float
    path: str = ""

    @property
    def dynamic_mw(self) -> float:
        return self.internal_mw + self.switching_mw


@dataclass
class Sample:
    name: str
    split: str
    features: dict[str, float]
    group_dynamic_mw: np.ndarray
    group_leakage_mw: np.ndarray
    true_dynamic_mw: float
    true_leakage_mw: float
    true_total_mw: float
    report_path: Path


@dataclass
class FittedModel:
    feature_order: list[str]
    leak_feature_order: list[str]
    row_groups: list[str]
    B_mw_per_count: np.ndarray
    alpha_mw_per_count: np.ndarray
    W: np.ndarray
    leakage_theta_mw_per_count: np.ndarray
    proxy_weights: dict[str, float]
    proxy_a: float
    proxy_b: float
    fitted_gamma: dict[str, float]
    training_names: list[str]

    def base_dynamic_mw(self, features: dict[str, float]) -> float:
        x = feature_vector(features, self.feature_order)
        return float(np.sum(self.B_mw_per_count @ x))

    def gamma_hat(self, features: dict[str, float]) -> float:
        proxy = activity_proxy(features, self.proxy_weights)
        return float(math.exp(self.proxy_a * math.log(max(proxy, EPS)) + self.proxy_b))

    def leakage_mw(self, features: dict[str, float]) -> float:
        z = feature_vector(features, self.leak_feature_order)
        return float(np.dot(self.leakage_theta_mw_per_count, z))

    def predict_total_mw(self, features: dict[str, float]) -> float:
        return self.base_dynamic_mw(features) * self.gamma_hat(features) + self.leakage_mw(features)

    def predict_group_total_mw(self, features: dict[str, float]) -> np.ndarray:
        x = feature_vector(features, self.feature_order)
        dynamic = (self.B_mw_per_count @ x) * self.gamma_hat(features)
        leakage = np.zeros(len(self.row_groups), dtype=float)
        for index, feature in enumerate(self.leak_feature_order):
            group = FEATURE_GROUP.get(feature, "OTHER")
            group_index = self.row_groups.index(group)
            leakage[group_index] += (
                self.leakage_theta_mw_per_count[index]
                * float(features.get(feature, 0.0))
            )
        return dynamic + leakage

    def oracle_total_mw(self, sample: Sample) -> float:
        gamma = self.fitted_gamma[sample.name]
        return self.base_dynamic_mw(sample.features) * gamma + self.leakage_mw(sample.features)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unit_scale_to_mw(unit: str) -> float:
    if unit == "W":
        return 1000.0
    if unit == "mW":
        return 1.0
    raise ValueError(f"power_unit must be exactly 'W' or 'mW', not {unit!r}.")


def extract_hierarchy_lines(lines: list[str]) -> list[str]:
    header_index = None
    separator_index = None
    for index, line in enumerate(lines):
        if "Hierarchy" in line and "Power" in line:
            header_index = index
            break
    if header_index is None:
        raise ValueError("Could not find a PTPX hierarchy-power table header.")

    for index in range(header_index, min(header_index + 80, len(lines))):
        stripped = lines[index].strip()
        if len(stripped) >= 10 and set(stripped) <= {"-"}:
            separator_index = index
            break
    if separator_index is None:
        raise ValueError("Could not find the hierarchy-table separator.")

    result = []
    for line in lines[separator_index + 1 :]:
        if not line.strip():
            break
        result.append(line.rstrip("\n"))
    if not result:
        raise ValueError("The hierarchy table contains no data rows.")
    return result


def parse_hierarchy_rows(lines: list[str], power_unit: str) -> list[HierRow]:
    scale = unit_scale_to_mw(power_unit)
    rows = []
    for raw in extract_hierarchy_lines(lines):
        match = LINE_RE.match(raw)
        if not match:
            continue
        rows.append(
            HierRow(
                name=match.group("name").strip(),
                indent=len(match.group("indent")),
                internal_mw=float(match.group("internal")) * scale,
                switching_mw=float(match.group("switching")) * scale,
                leakage_mw=float(match.group("leakage")) * scale,
                total_mw=float(match.group("total")) * scale,
            )
        )
    if not rows:
        raise ValueError("No hierarchy rows matched the expected PTPX numeric format.")
    return rows


def choose_root_index(rows: list[HierRow], candidates: Sequence[str]) -> int:
    for candidate in candidates:
        matches = [
            (index, row)
            for index, row in enumerate(rows)
            if candidate in row.name
        ]
        if matches:
            matches.sort(key=lambda item: (len(item[1].name), -item[1].dynamic_mw))
            return matches[0][0]
    raise ValueError(
        "None of the configured subtree candidates appears in the hierarchy table: "
        + ", ".join(candidates)
    )


def subtree_with_paths(rows: list[HierRow], root_index: int) -> tuple[HierRow, list[HierRow]]:
    root = rows[root_index]
    descendants = []
    stack: list[HierRow] = [root]
    for row in rows[root_index + 1 :]:
        if row.indent <= root.indent:
            break
        while stack and stack[-1].indent >= row.indent:
            stack.pop()
        path = "/".join([item.name for item in stack] + [row.name])
        descendants.append(
            HierRow(
                name=row.name,
                indent=row.indent,
                internal_mw=row.internal_mw,
                switching_mw=row.switching_mw,
                leakage_mw=row.leakage_mw,
                total_mw=row.total_mw,
                path=path,
            )
        )
        stack.append(row)
    if not descendants:
        raise ValueError(f"Selected subtree root {root.name!r} has no descendants.")
    return root, descendants


def select_nonoverlapping_rows(
    root: HierRow,
    descendants: list[HierRow],
    hierarchy_mode: str,
    indent_delta: int | None,
) -> list[HierRow]:
    if hierarchy_mode == "direct_children":
        child_indent = min(row.indent for row in descendants)
        selected = [row for row in descendants if row.indent == child_indent]
    elif hierarchy_mode == "leaf":
        selected = []
        for index, row in enumerate(descendants):
            next_is_child = (
                index + 1 < len(descendants)
                and descendants[index + 1].indent > row.indent
            )
            if not next_is_child:
                selected.append(row)
    elif hierarchy_mode == "indent_delta":
        if indent_delta is None or indent_delta <= 0:
            raise ValueError("indent_delta mode requires a positive indent_delta.")
        selected = [
            row for row in descendants if row.indent - root.indent == indent_delta
        ]
    else:
        raise ValueError(
            "hierarchy_mode must be direct_children, leaf, or indent_delta."
        )
    if not selected:
        raise ValueError(
            f"hierarchy_mode={hierarchy_mode!r} selected no rows below {root.name!r}."
        )
    return selected


def classify_group(path: str) -> str:
    text = path.lower()
    if "sb_route" in text:
        return "SB"
    if "rmux_route" in text:
        return "RMUX"
    if "reg_route" in text:
        return "REG"
    if "port_route" in text:
        return "PORT"
    if "pipeline_reg" in text or re.search(r"\breg_r\d+\b", text):
        return "PIPE"
    if "pe_tile" in text:
        return "PE"
    if "mem_tile" in text:
        return "MEM"
    if "io_tile" in text:
        return "IO"
    return "OTHER"


def parse_report(
    path: Path,
    power_unit: str,
    root_candidates: Sequence[str],
    hierarchy_mode: str,
    indent_delta: int | None,
    consistency_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = parse_hierarchy_rows(lines, power_unit)
    root_index = choose_root_index(rows, root_candidates)
    root, descendants = subtree_with_paths(rows, root_index)
    selected = select_nonoverlapping_rows(
        root, descendants, hierarchy_mode, indent_delta
    )

    grouped_dynamic = {group: 0.0 for group in ROW_GROUPS}
    grouped_leakage = {group: 0.0 for group in ROW_GROUPS}
    for row in selected:
        group = classify_group(row.path)
        grouped_dynamic[group] += row.dynamic_mw
        grouped_leakage[group] += row.leakage_mw

    selected_dynamic = sum(grouped_dynamic.values())
    residual_dynamic = root.dynamic_mw - selected_dynamic
    allowed_error = max(consistency_tolerance * max(root.dynamic_mw, 1.0), 1e-6)
    if residual_dynamic < -allowed_error:
        raise ValueError(
            f"{path}: selected hierarchy rows sum to {selected_dynamic:.6f} mW, "
            f"which exceeds root dynamic power {root.dynamic_mw:.6f} mW. "
            "The selected rows overlap. Change hierarchy_mode or indent_delta."
        )
    grouped_dynamic["SELF"] = max(0.0, residual_dynamic)

    selected_leakage = sum(grouped_leakage.values())
    residual_leakage = root.leakage_mw - selected_leakage
    leakage_error = max(
        consistency_tolerance * max(root.leakage_mw, 1.0), 1e-6
    )
    if residual_leakage < -leakage_error:
        raise ValueError(
            f"{path}: selected hierarchy rows sum to {selected_leakage:.6f} mW "
            f"of leakage, which exceeds root leakage {root.leakage_mw:.6f} mW."
        )
    grouped_leakage["SELF"] = max(0.0, residual_leakage)

    reconstructed_dynamic = sum(grouped_dynamic.values())
    if abs(reconstructed_dynamic - root.dynamic_mw) > allowed_error:
        raise ValueError(
            f"{path}: grouped dynamic power does not reconstruct the root."
        )
    reconstructed_leakage = sum(grouped_leakage.values())
    if abs(reconstructed_leakage - root.leakage_mw) > leakage_error:
        raise ValueError(
            f"{path}: grouped leakage power does not reconstruct the root."
        )
    if abs((root.dynamic_mw + root.leakage_mw) - root.total_mw) > allowed_error:
        raise ValueError(
            f"{path}: root internal + switching + leakage does not equal total "
            "within the configured tolerance."
        )

    dynamic_vector = np.array(
        [grouped_dynamic[group] for group in ROW_GROUPS], dtype=float
    )
    leakage_vector = np.array(
        [grouped_leakage[group] for group in ROW_GROUPS], dtype=float
    )
    return (
        dynamic_vector,
        leakage_vector,
        root.dynamic_mw,
        root.leakage_mw,
        root.total_mw,
    )


def feature_vector(features: dict[str, float], order: Sequence[str]) -> np.ndarray:
    return np.array([float(features.get(name, 0.0)) for name in order], dtype=float)


def load_samples(
    manifest_path: Path,
    allow_private_inputs: bool,
) -> tuple[dict[str, Any], list[Sample]]:
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path}: unsupported schema_version "
            f"{manifest.get('schema_version')!r}."
        )
    is_synthetic = bool(manifest.get("is_synthetic", False))
    if not is_synthetic and not allow_private_inputs:
        raise PermissionError(
            "This manifest is not marked synthetic. Private report processing "
            "requires --allow-private-inputs. Keep its outputs out of the public repository."
        )

    power_unit = str(manifest.get("power_unit", ""))
    unit_scale_to_mw(power_unit)
    candidates = [str(value) for value in manifest.get("root_candidates", [])]
    if not candidates:
        raise ValueError("manifest.root_candidates must not be empty.")
    hierarchy_mode = str(manifest.get("hierarchy_mode", "leaf"))
    indent_delta_value = manifest.get("indent_delta")
    indent_delta = (
        None if indent_delta_value is None else int(indent_delta_value)
    )
    tolerance = float(manifest.get("consistency_tolerance", 1e-4))

    dataset_rows = manifest.get("datasets")
    if not isinstance(dataset_rows, list) or not dataset_rows:
        raise ValueError("manifest.datasets must be a nonempty list.")

    samples = []
    seen = set()
    for item in dataset_rows:
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"Duplicate dataset name in manifest: {name}")
        seen.add(name)
        split = str(item["split"])
        if split not in {"train", "heldout"}:
            raise ValueError(f"{name}: split must be train or heldout.")
        report_path = (manifest_path.parent / str(item["report"])).resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"{name}: report does not exist: {report_path}")
        features = {key: float(value) for key, value in item["features"].items()}
        features["bias"] = 1.0
        missing = [
            feature
            for feature in FEATURE_ORDER
            if feature != "bias" and feature not in features
        ]
        if missing:
            raise ValueError(f"{name}: missing features: {', '.join(missing)}")

        group_dynamic, group_leakage, dynamic, leakage, total = parse_report(
            report_path,
            power_unit=power_unit,
            root_candidates=candidates,
            hierarchy_mode=hierarchy_mode,
            indent_delta=indent_delta,
            consistency_tolerance=tolerance,
        )
        samples.append(
            Sample(
                name=name,
                split=split,
                features=features,
                group_dynamic_mw=group_dynamic,
                group_leakage_mw=group_leakage,
                true_dynamic_mw=dynamic,
                true_leakage_mw=leakage,
                true_total_mw=total,
                report_path=report_path,
            )
        )
    if not any(sample.split == "train" for sample in samples):
        raise ValueError("The manifest contains no training samples.")
    return manifest, samples


def nonnegative_ridge(
    design: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """
    Solve min ||A x - y||^2 + lambda ||x||^2 subject to x >= 0.
    This uses an augmented least-squares system and a bounded optimizer.
    """
    if design.ndim != 2:
        raise ValueError("design must be a 2D matrix.")
    if target.shape != (design.shape[0],):
        raise ValueError("target length does not match design rows.")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be nonnegative.")
    n_coeffs = design.shape[1]
    if n_coeffs == 0:
        return np.zeros(0, dtype=float)
    if ridge_lambda > 0:
        augmented_design = np.vstack(
            [design, math.sqrt(ridge_lambda) * np.eye(n_coeffs)]
        )
        augmented_target = np.concatenate([target, np.zeros(n_coeffs)])
    else:
        augmented_design = design
        augmented_target = target
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
        max_iter=5000,
    )
    if not result.success:
        raise RuntimeError(f"Nonnegative ridge fit failed: {result.message}")
    return result.x


def event_mask(feature_order: Sequence[str]) -> np.ndarray:
    mask = np.zeros((len(ROW_GROUPS), len(feature_order)), dtype=bool)
    for column, feature in enumerate(feature_order):
        if feature == "bias":
            mask[:, column] = True
        else:
            group = FEATURE_GROUP.get(feature, "OTHER")
            mask[ROW_GROUPS.index(group), column] = True
    return mask


def activity_proxy(
    features: dict[str, float],
    proxy_weights: dict[str, float],
) -> float:
    value = sum(
        float(weight) * float(features.get(feature, 0.0))
        for feature, weight in proxy_weights.items()
    )
    return max(value, EPS)


def fit_log_proxy(
    samples: Sequence[Sample],
    gamma: np.ndarray,
    proxy_weights: dict[str, float],
) -> tuple[float, float]:
    if len(samples) == 1:
        return 0.0, float(math.log(max(float(gamma[0]), EPS)))
    x = np.log(
        np.array(
            [activity_proxy(sample.features, proxy_weights) for sample in samples],
            dtype=float,
        )
    )
    y = np.log(np.maximum(gamma, EPS))
    design = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(a), float(b)


def fit_hierarchical_model(
    samples: Sequence[Sample],
    feature_order: list[str],
    leak_feature_order: list[str],
    proxy_weights: dict[str, float],
    dynamic_ridge: float,
    leakage_ridge: float,
    gamma_iterations: int,
    gamma_prior_strength: float,
) -> FittedModel:
    if not samples:
        raise ValueError("Cannot fit a model with no samples.")

    X = np.column_stack(
        [feature_vector(sample.features, feature_order) for sample in samples]
    )
    Y = np.column_stack([sample.group_dynamic_mw for sample in samples])
    true_dynamic = np.array([sample.true_dynamic_mw for sample in samples])
    mask = event_mask(feature_order)
    gamma = np.ones(len(samples), dtype=float)
    B = np.zeros((len(ROW_GROUPS), len(feature_order)), dtype=float)

    # Alternating fit. Every conditional coefficient update is a true
    # nonnegative ridge problem. The median-gamma normalization resolves the
    # scale ambiguity between B and gamma.
    for _ in range(max(1, gamma_iterations)):
        adjusted_targets = Y / np.maximum(gamma[None, :], EPS)
        for row_index in range(len(ROW_GROUPS)):
            allowed = np.flatnonzero(mask[row_index])
            design = X[allowed, :].T
            target = adjusted_targets[row_index, :]
            coeffs = nonnegative_ridge(design, target, dynamic_ridge)
            B[row_index, :] = 0.0
            B[row_index, allowed] = coeffs

        base_dynamic = np.sum(B @ X, axis=0)
        raw_gamma = true_dynamic / np.maximum(base_dynamic, EPS)
        raw_gamma = np.clip(raw_gamma, 0.2, 5.0)
        gamma = (
            raw_gamma + gamma_prior_strength * np.ones_like(raw_gamma)
        ) / (1.0 + gamma_prior_strength)
        gamma /= max(float(np.median(gamma)), EPS)

    # Refit B once using the final normalized gamma.
    adjusted_targets = Y / np.maximum(gamma[None, :], EPS)
    for row_index in range(len(ROW_GROUPS)):
        allowed = np.flatnonzero(mask[row_index])
        B[row_index, :] = 0.0
        B[row_index, allowed] = nonnegative_ridge(
            X[allowed, :].T,
            adjusted_targets[row_index, :],
            dynamic_ridge,
        )

    alpha = np.sum(B, axis=0)
    W = np.divide(
        B,
        alpha[None, :],
        out=np.zeros_like(B),
        where=alpha[None, :] > EPS,
    )

    Z = np.column_stack(
        [feature_vector(sample.features, leak_feature_order) for sample in samples]
    )
    leakage_target = np.array([sample.true_leakage_mw for sample in samples])
    leakage_theta = nonnegative_ridge(
        Z.T, leakage_target, leakage_ridge
    )

    proxy_a, proxy_b = fit_log_proxy(samples, gamma, proxy_weights)
    return FittedModel(
        feature_order=feature_order,
        leak_feature_order=leak_feature_order,
        row_groups=list(ROW_GROUPS),
        B_mw_per_count=B,
        alpha_mw_per_count=alpha,
        W=W,
        leakage_theta_mw_per_count=leakage_theta,
        proxy_weights=proxy_weights,
        proxy_a=proxy_a,
        proxy_b=proxy_b,
        fitted_gamma={
            sample.name: float(gamma[index])
            for index, sample in enumerate(samples)
        },
        training_names=[sample.name for sample in samples],
    )


def ape_percent(true_value: float, prediction: float) -> float:
    return 100.0 * abs(prediction - true_value) / max(abs(true_value), EPS)


def metrics(
    true_values: Sequence[float],
    predictions: Sequence[float],
) -> dict[str, float]:
    true_array = np.asarray(true_values, dtype=float)
    pred_array = np.asarray(predictions, dtype=float)
    residual = pred_array - true_array
    mape = float(
        100.0
        * np.mean(np.abs(residual) / np.maximum(np.abs(true_array), EPS))
    )
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((true_array - np.mean(true_array)) ** 2))
    r2 = float("nan") if ss_tot <= EPS else 1.0 - ss_res / ss_tot
    slope = float(np.dot(true_array, pred_array) / max(np.dot(true_array, true_array), EPS))
    return {
        "MAE_mW": mae,
        "MAPE_percent": mape,
        "R2": r2,
        "slope_through_origin": slope,
    }


def model_to_json(model: FittedModel, is_synthetic: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": is_synthetic,
        "units": {
            "power": "mW",
            "dynamic_coefficients": "mW/count",
            "leakage_coefficients": "mW/count",
        },
        "training_names": model.training_names,
        "feature_order": model.feature_order,
        "row_groups": model.row_groups,
        "B_mw_per_count": model.B_mw_per_count.tolist(),
        "alpha_mw_per_count": {
            name: float(model.alpha_mw_per_count[index])
            for index, name in enumerate(model.feature_order)
        },
        "W": model.W.tolist(),
        "activity_proxy": {
            "weights": model.proxy_weights,
            "log_fit_a": model.proxy_a,
            "log_fit_b": model.proxy_b,
        },
        "leakage": {
            "feature_order": model.leak_feature_order,
            "theta_mw_per_count": {
                name: float(model.leakage_theta_mw_per_count[index])
                for index, name in enumerate(model.leak_feature_order)
            },
        },
        "fitted_gamma": model.fitted_gamma,
    }


def collapse_figure10_groups(values: np.ndarray) -> dict[str, float]:
    if values.shape != (len(ROW_GROUPS),):
        raise ValueError("Figure 10 group vector has the wrong shape.")
    collapsed = {category: 0.0 for category in FIGURE10_CATEGORIES}
    for index, group in enumerate(ROW_GROUPS):
        target = "REG" if group == "PIPE" else "OTHER" if group == "SELF" else group
        collapsed[target] += float(values[index])
    return collapsed


def figure10_mapping_rows(model: FittedModel) -> list[dict[str, Any]]:
    specifications = [
        (
            "PE tile",
            ["num_pe_tiles", "num_pe_ports"],
            [r"$\beta_{\mathrm{pe\_tiles}}$", r"$\beta_{\mathrm{pe\_ports}}$"],
        ),
        (
            "MEM tile",
            ["num_mem_tiles", "num_mem_ports"],
            [r"$\beta_{\mathrm{mem\_tiles}}$", r"$\beta_{\mathrm{mem\_ports}}$"],
        ),
        (
            "REG",
            ["num_ic_reg", "num_pipeline_regs"],
            [r"$\beta_{\mathrm{registers}}$", r"$\beta_{\mathrm{pipeline}}$"],
        ),
        (
            "IO tile",
            ["num_io_tiles"],
            [r"$\beta_{\mathrm{IO\_tiles}}$"],
        ),
        (
            "IC (SB)",
            ["num_ic_sb"],
            [r"$\beta_{\mathrm{ic\_sb}}$"],
        ),
        (
            "IC (RMUX)",
            ["num_ic_rmux"],
            [r"$\beta_{\mathrm{ic\_rmux}}$"],
        ),
        (
            "IC (PORT)",
            ["num_ic_port"],
            [r"$\beta_{\mathrm{ic\_port}}$"],
        ),
    ]
    rows = []
    for primitive, features, symbols in specifications:
        feature_indices = [model.feature_order.index(feature) for feature in features]
        top_groups = []
        coefficient_values = []
        for feature, index in zip(features, feature_indices):
            top_group = model.row_groups[int(np.argmax(model.W[:, index]))]
            if top_group == "PIPE":
                top_group = "REG"
            if top_group == "SELF":
                top_group = "OTHER"
            if top_group not in top_groups:
                top_groups.append(top_group)
            coefficient_values.append(float(model.alpha_mw_per_count[index]))
        event_labels = [
            feature.removeprefix("num_")
            .replace("ic_reg", "registers")
            .replace("pipeline_regs", "pipeline_regs")
            for feature in features
        ]
        rows.append(
            {
                "primitive": primitive,
                "events": ", ".join(event_labels),
                "coefficient_symbols": ", ".join(symbols),
                "coefficients_mW_per_count": ", ".join(
                    f"{value:.6g}" for value in coefficient_values
                ),
                "top_ptpx_row": ", ".join(top_groups),
            }
        )
    return rows


def make_plots(
    figure5_rows: list[dict[str, Any]],
    figure9_rows: list[dict[str, Any]],
    figure10_mapping: list[dict[str, Any]],
    figure10_breakdown: list[dict[str, Any]],
    output_dir: Path,
    is_synthetic: bool,
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "capstone-matplotlib-cache"),
    )
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    title_suffix = " (synthetic demonstration)" if is_synthetic else ""

    short_name = {
        "vec_elemadd": "VecAdd",
        "mat_elemmul": "ElemMul",
        "tensor3_ttv": "TTV",
        "tensor3_mttkrp": "MTTKRP",
        "tensor3_innerprod": "T3-Inner",
        "mat_mask_tri": "TriMask",
        "mat_sddmm": "SDDMM",
        "mat_mattransmul": "MatT-Mul",
        "gaussian": "Gaussian",
        "harris": "Harris",
        "unsharp": "Unsharp",
    }

    # Figure 5: fitting coverage (all-kernel vs. VecAdd-only fitting).
    # Match the camera-ready dumbbell-plot design while using the bundled
    # synthetic values in this workflow.
    gray = "#7A7A7A"
    grid = "#D8D8D8"
    row_band = "#F6F6F6"
    blue = "#EBF6FE"
    pink = "#F8DADA"
    magenta = "#D81B8A"

    figure5_rows_sorted = sorted(
        figure5_rows,
        key=lambda row: float(row["percent_difference_all"]),
    )
    all_values = np.array(
        [float(row["percent_difference_all"]) for row in figure5_rows_sorted],
        dtype=float,
    )
    ref_values = np.array(
        [float(row["percent_difference_reference"]) for row in figure5_rows_sorted],
        dtype=float,
    )
    calibration_gap = np.abs(ref_values - all_values)
    highlight_rows = set(np.argsort(calibration_gap)[-3:])
    y = np.arange(len(figure5_rows_sorted))

    fig5, ax5 = plt.subplots(figsize=(7.6, 3.8), dpi=200)
    for row in y:
        if row % 2 == 0:
            ax5.axhspan(row - 0.5, row + 0.5, color=row_band, zorder=0)

    for row in y:
        highlighted = row in highlight_rows
        ax5.plot(
            [all_values[row], ref_values[row]],
            [row, row],
            color=magenta if highlighted else gray,
            linewidth=2.1 if highlighted else 1.6,
            alpha=0.95 if highlighted else 0.82,
            solid_capstyle="round",
            zorder=2,
        )

    ax5.scatter(
        all_values, y, s=76, marker="o", facecolor=blue, edgecolor="black",
        linewidth=1.15, label="All-kernel fitting", zorder=4,
    )
    ax5.scatter(
        ref_values, y, s=76, marker="s", facecolor=pink, edgecolor="black",
        linewidth=1.15, label="VecAdd-only fitting", zorder=4,
    )

    ax5.set_ylim(len(y) - 0.45, -1.0)
    ax5.set_yticks(y)
    ax5.set_yticklabels(
        [
            short_name.get(str(row["workload"]), str(row["workload"]))
            for row in figure5_rows_sorted
        ],
        fontsize=13.5,
    )

    data_max = float(max(all_values.max(), ref_values.max()))
    x_upper = max(10.0, float(np.ceil((data_max + 5.0) / 5.0) * 5.0))
    ax5.set_xlim(-0.8, x_upper)
    ax5.xaxis.set_major_locator(MultipleLocator(5))
    ax5.xaxis.set_minor_locator(MultipleLocator(2.5))

    for row in sorted(highlight_rows):
        right_point = max(all_values[row], ref_values[row])
        ax5.text(
            right_point + 0.65, row, rf"$\Delta$ {calibration_gap[row]:.1f}",
            color=magenta, fontsize=10.5, ha="left", va="center", zorder=5,
        )

    ax5.annotate(
        "Lower error",
        xy=(0.7, -0.67),
        xytext=(10.0, -0.67),
        color=magenta,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": magenta,
            "linewidth": 1.25,
            "shrinkA": 3,
            "shrinkB": 2,
        },
    )

    ax5.set_xlabel("Percent Difference (%)", fontsize=16, labelpad=7)
    ax5.set_axisbelow(True)
    ax5.grid(axis="x", which="major", color=grid, linewidth=0.7, alpha=0.8)
    ax5.tick_params(axis="x", which="major", labelsize=12.5, length=5, width=1.0)
    ax5.tick_params(axis="x", which="minor", length=2.8, width=0.8)
    ax5.tick_params(axis="y", which="major", length=0, pad=7)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["left"].set_linewidth(1.0)
    ax5.spines["bottom"].set_linewidth(1.0)

    leg5 = ax5.legend(
        loc="upper right", bbox_to_anchor=(0.988, 0.985), ncol=1,
        frameon=True, fancybox=True, fontsize=13.5, handlelength=1.8,
        handletextpad=0.6, labelspacing=0.45, borderpad=0.45,
    )
    leg5.get_frame().set_facecolor("white")
    leg5.get_frame().set_alpha(0.75)
    leg5.get_frame().set_edgecolor("black")
    leg5.get_frame().set_linewidth(1.2)
    fig5.subplots_adjust(left=0.16, right=0.995, bottom=0.19, top=0.985)

    for suffix in ("pdf", "png"):
        fig5.savefig(
            output_dir / f"figure5_fitting_scope.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            dpi=600 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig5)

    # Figure 9: one-panel predicted-vs-signoff accuracy plot.
    signoff = np.array([float(row["signoff_total_mW"]) for row in figure9_rows])
    capstone = np.array([float(row["capstone_prediction_mW"]) for row in figure9_rows])
    oracle = np.array([float(row["oracle_prediction_mW"]) for row in figure9_rows])
    is_train = np.array([row["split"] == "train" for row in figure9_rows], dtype=bool)

    def fit_metrics(xvals: np.ndarray, yvals: np.ndarray) -> tuple[float, float, float, float]:
        slope, intercept = np.polyfit(xvals, yvals, 1)
        yfit = slope * xvals + intercept
        ss_res = float(np.sum((yvals - yfit) ** 2))
        ss_tot = float(np.sum((yvals - np.mean(yvals)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mape = float(np.mean(np.abs((yvals - xvals) / xvals)) * 100.0)
        return float(slope), float(intercept), float(r2), mape

    s_cap, b_cap, r2_cap, mape_cap = fit_metrics(signoff[is_train], capstone[is_train])
    s_ora, b_ora, r2_ora, mape_ora = fit_metrics(signoff[is_train], oracle[is_train])

    pad = 0.08 * max(float(np.max(signoff)), float(np.max(capstone)), float(np.max(oracle)))
    low = max(0.0, min(float(np.min(signoff)), float(np.min(capstone)), float(np.min(oracle))) - pad)
    high = max(float(np.max(signoff)), float(np.max(capstone)), float(np.max(oracle))) + pad
    xx = np.array([low, high], dtype=float)

    fig9, ax9 = plt.subplots(figsize=(7.25, 3.95), dpi=200)
    for i in range(len(figure9_rows)):
        ax9.plot(
            [signoff[i], signoff[i]],
            [capstone[i], oracle[i]],
            color="#8A8A8A",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )
    ax9.plot(xx, xx, "--", color="#8A8A8A", linewidth=1.2, label="y = x", zorder=0)
    ax9.plot(xx, s_cap * xx + b_cap, color="black", linewidth=1.55, label="Capstone fit", zorder=1)
    ax9.plot(xx, s_ora * xx + b_ora, color="#7C9750", linewidth=1.35, linestyle="-.", label="Oracle fit", zorder=1)
    ax9.scatter(
        signoff[is_train], capstone[is_train], s=55, marker="o",
        facecolor="#EAF4FB", edgecolor="black", linewidth=0.9,
        zorder=3, label="Capstone (in-sample)"
    )
    ax9.scatter(
        signoff[~is_train], capstone[~is_train], s=60, marker="s",
        facecolor="#F8DADA", edgecolor="black", linewidth=0.9,
        zorder=3, label="Capstone (held-out)"
    )
    ax9.scatter(
        signoff, oracle, s=44, marker="D", facecolor="#F0FEDB",
        edgecolor="black", linewidth=0.85, zorder=4, label="Oracle"
    )

    pct = 100.0 * np.abs(capstone - signoff) / signoff
    label_offsets = {
        "vec_elemadd": (0, 12),
        "tensor3_ttv": (-13, 12),
        "mat_elemmul": (15, -14),
        "mat_mask_tri": (0, 12),
        "mat_sddmm": (12, 10),
        "mat_mattransmul": (-5, 12),
        "tensor3_mttkrp": (-14, 12),
        "tensor3_innerprod": (0, -14),
        "harris": (11, -14),
        "unsharp": (-10, 11),
        "gaussian": (-10, -14),
    }
    for i, row in enumerate(figure9_rows):
        dx, dy = label_offsets.get(str(row["workload"]), (0, 10))
        label = f"{short_name.get(str(row['workload']), str(row['workload']))}\n{pct[i]:.1f}%"
        ax9.annotate(
            label,
            xy=(signoff[i], capstone[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=8.3,
            linespacing=0.88,
            bbox=dict(boxstyle="round,pad=0.07", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=5,
        )

    ax9.set_xlabel("Signoff Power (mW)", fontsize=12.5)
    ax9.set_ylabel("Predicted Power (mW)", fontsize=12.5)
    ax9.set_xlim(low, high)
    ax9.set_ylim(low, high)
    ax9.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    ax9.set_axisbelow(True)
    ax9.spines["top"].set_visible(False)
    ax9.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=7,
               markerfacecolor="#EAF4FB", markeredgecolor="black", label="Capstone (in-sample)"),
        Line2D([], [], linestyle="none", marker="s", markersize=7,
               markerfacecolor="#F8DADA", markeredgecolor="black", label="Capstone (held-out)"),
        Line2D([], [], linestyle="none", marker="D", markersize=6,
               markerfacecolor="#F0FEDB", markeredgecolor="black", label="Oracle"),
        Line2D([], [], linestyle="--", color="#8A8A8A", linewidth=1.2, label="y = x"),
        Line2D([], [], linestyle="-", color="black", linewidth=1.55, label="Capstone fit"),
        Line2D([], [], linestyle="-.", color="#7C9750", linewidth=1.35, label="Oracle fit"),
    ]
    leg9 = ax9.legend(
        handles=legend_handles, loc="upper left", frameon=True, fancybox=True,
        fontsize=8.8, borderpad=0.42, handlelength=1.55, handletextpad=0.45,
        labelspacing=0.20,
    )
    leg9.get_frame().set_facecolor("white")
    leg9.get_frame().set_alpha(0.84)
    leg9.get_frame().set_edgecolor("black")

    ax9.text(
        0.985, 0.035,
        "Fit/metrics on in-sample kernels:\n"
        + rf"Capstone: slope = {s_cap:.2f}, R$^2$ = {r2_cap:.3f}, MAPE = {mape_cap:.1f}%"
        + "\n"
        + rf"Oracle: slope = {s_ora:.2f}, R$^2$ = {r2_ora:.3f}, MAPE = {mape_ora:.1f}%",
        transform=ax9.transAxes, ha="right", va="bottom", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor="none", alpha=0.84),
    )
    fig9.tight_layout(pad=0.25)
    for suffix in ("pdf", "png"):
        fig9.savefig(
            output_dir / f"figure9_power_accuracy.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            dpi=600 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig9)

    # Figure 10: learned mapping and selected-workload power breakdown.
    if not figure10_breakdown:
        raise ValueError("Figure 10 breakdown contains no rows.")
    figure10_workload = str(figure10_breakdown[0]["workload"])
    fig10, (ax10a, ax10b) = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.45),
        dpi=200,
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    ax10a.axis("off")
    table = ax10a.table(
        cellText=[
            [
                row["primitive"],
                row["events"],
                row["coefficient_symbols"],
                row["top_ptpx_row"],
            ]
            for row in figure10_mapping
        ],
        colLabels=["Primitive", "Events", "Coeffs", "Top PTPX Row"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.19, 0.31, 0.30, 0.20],
        bbox=[0.0, 0.16, 1.0, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.PAD = 0.025
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row_index == len(figure10_mapping):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
    ax10a.text(
        0.0,
        0.07,
        "(a) Event-to-PTPX mapping.",
        transform=ax10a.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    category_colors = {
        "PE": "#e9f5fc",
        "MEM": "#f7d9d9",
        "IO": "#effbd9",
        "SB": "#fff8d5",
        "RMUX": "#f1f1f1",
        "REG": "#f6e7d8",
        "PORT": "#e6e6e6",
        "OTHER": "#fafafa",
    }
    bottoms = np.zeros(2, dtype=float)
    for category in FIGURE10_CATEGORIES:
        matching = [
            row for row in figure10_breakdown if row["category"] == category
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Figure 10 needs exactly one {category} breakdown row."
            )
        row = matching[0]
        values = np.array(
            [float(row["ptpx_mW"]), float(row["model_mW"])], dtype=float
        )
        ax10b.bar(
            [0, 1],
            values,
            bottom=bottoms,
            width=0.36,
            color=category_colors[category],
            edgecolor="black",
            linewidth=0.6,
            label=category,
        )
        bottoms += values
    ax10b.set_xticks([0, 1], ["Synthetic PTPX" if is_synthetic else "PTPX", "Model"])
    ax10b.set_ylabel("Power (mW)")
    ax10b.spines["top"].set_visible(False)
    ax10b.spines["right"].set_visible(False)
    ax10b.grid(axis="y", alpha=0.3)
    ax10b.set_axisbelow(True)
    ax10b.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fontsize=6.7,
        columnspacing=0.6,
        handlelength=1.0,
    )
    ax10b.text(
        0.5,
        -0.18,
        "(b) Power breakdown.",
        transform=ax10b.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    fig10.subplots_adjust(
        left=0.03, right=0.995, bottom=0.16, top=0.93, wspace=0.14
    )
    for suffix in ("pdf", "png"):
        fig10.savefig(
            output_dir / f"figure10_mapping_and_breakdown.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig10)


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    manifest, samples = load_samples(
        manifest_path, allow_private_inputs=args.allow_private_inputs
    )
    is_synthetic = bool(manifest["is_synthetic"])

    proxy_weights = {
        key: float(value)
        for key, value in manifest.get(
            "proxy_weights", DEFAULT_PROXY_WEIGHTS
        ).items()
    }
    feature_order = [str(value) for value in manifest.get("feature_order", FEATURE_ORDER)]
    leak_feature_order = [
        str(value)
        for value in manifest.get("leak_feature_order", LEAK_FEATURE_ORDER)
    ]

    fit_kwargs = {
        "feature_order": feature_order,
        "leak_feature_order": leak_feature_order,
        "proxy_weights": proxy_weights,
        "dynamic_ridge": args.dynamic_ridge,
        "leakage_ridge": args.leakage_ridge,
        "gamma_iterations": args.gamma_iterations,
        "gamma_prior_strength": args.gamma_prior_strength,
    }

    training_samples = [sample for sample in samples if sample.split == "train"]
    reference_name = str(manifest.get("reference_calibration_workload", "vec_elemadd"))
    reference_samples = [sample for sample in samples if sample.name == reference_name]
    if len(reference_samples) != 1:
        raise ValueError(
            "reference_calibration_workload must identify exactly one dataset."
        )

    primary_model = fit_hierarchical_model(training_samples, **fit_kwargs)
    all_model = fit_hierarchical_model(samples, **fit_kwargs)
    oracle_model = all_model

    # Figure 5 follows the camera-ready plotting source.  The all-kernel
    # model supplies the base predictions, and the VecAdd-only comparison
    # uses one scale factor chosen so that the VecAdd prediction matches its
    # reference power exactly.  This mirrors the released Figure 5 script
    # while keeping the public synthetic workflow independent of private
    # row-level PTPX reports.
    all_predictions = {
        sample.name: all_model.predict_total_mw(sample.features)
        for sample in samples
    }
    reference_sample = reference_samples[0]
    reference_base = all_predictions[reference_name]
    if abs(reference_base) <= 1e-12:
        raise ValueError("Reference-workload prediction is zero; cannot construct Figure 5 scale factor.")
    vecadd_only_scale = reference_sample.true_total_mw / reference_base

    figure5_rows = []
    for sample in samples:
        all_prediction = all_predictions[sample.name]
        reference_prediction = all_prediction * vecadd_only_scale
        figure5_rows.append(
            {
                "workload": sample.name,
                "reference_workload": reference_name,
                "signoff_total_mW": sample.true_total_mw,
                "all_calibration_prediction_mW": all_prediction,
                "reference_calibration_prediction_mW": reference_prediction,
                "percent_difference_all": ape_percent(
                    sample.true_total_mw, all_prediction
                ),
                "percent_difference_reference": ape_percent(
                    sample.true_total_mw, reference_prediction
                ),
                "vecadd_only_scale_factor": vecadd_only_scale,
                "is_synthetic": is_synthetic,
            }
        )

    figure9_rows = []
    for sample in samples:
        capstone_prediction = primary_model.predict_total_mw(sample.features)
        oracle_prediction = oracle_model.oracle_total_mw(sample)
        figure9_rows.append(
            {
                "workload": sample.name,
                "split": sample.split,
                "signoff_total_mW": sample.true_total_mw,
                "capstone_prediction_mW": capstone_prediction,
                "oracle_prediction_mW": oracle_prediction,
                "capstone_ape_percent": ape_percent(
                    sample.true_total_mw, capstone_prediction
                ),
                "oracle_ape_percent": ape_percent(
                    sample.true_total_mw, oracle_prediction
                ),
                "is_synthetic": is_synthetic,
            }
        )

    figure10_workload = str(manifest.get("figure10_workload", "vec_elemadd"))
    figure10_samples = [
        sample for sample in samples if sample.name == figure10_workload
    ]
    if len(figure10_samples) != 1:
        raise ValueError(
            "figure10_workload must identify exactly one dataset."
        )
    figure10_sample = figure10_samples[0]
    figure10_mapping = figure10_mapping_rows(primary_model)
    ptpx_breakdown = collapse_figure10_groups(
        figure10_sample.group_dynamic_mw
        + figure10_sample.group_leakage_mw
    )
    model_breakdown = collapse_figure10_groups(
        primary_model.predict_group_total_mw(figure10_sample.features)
    )
    ptpx_breakdown_total = sum(ptpx_breakdown.values())
    model_breakdown_total = sum(model_breakdown.values())
    model_prediction = primary_model.predict_total_mw(figure10_sample.features)
    if not math.isclose(
        ptpx_breakdown_total,
        figure10_sample.true_total_mw,
        rel_tol=1e-8,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Figure 10 PTPX category breakdown does not sum to the report total."
        )
    if not math.isclose(
        model_breakdown_total,
        model_prediction,
        rel_tol=1e-8,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Figure 10 model category breakdown does not sum to the prediction."
        )
    figure10_breakdown = [
        {
            "workload": figure10_workload,
            "category": category,
            "ptpx_mW": ptpx_breakdown[category],
            "model_mW": model_breakdown[category],
            "is_synthetic": is_synthetic,
        }
        for category in FIGURE10_CATEGORIES
    ]

    write_csv(
        output_dir / "figure5_fitting_scope.csv",
        figure5_rows,
        [
            "workload",
            "reference_workload",
            "signoff_total_mW",
            "all_calibration_prediction_mW",
            "reference_calibration_prediction_mW",
            "percent_difference_all",
            "percent_difference_reference",
            "vecadd_only_scale_factor",
            "is_synthetic",
        ],
    )
    write_csv(
        output_dir / "figure10_mapping.csv",
        figure10_mapping,
        [
            "primitive",
            "events",
            "coefficient_symbols",
            "coefficients_mW_per_count",
            "top_ptpx_row",
        ],
    )
    write_csv(
        output_dir / "figure10_breakdown.csv",
        figure10_breakdown,
        [
            "workload",
            "category",
            "ptpx_mW",
            "model_mW",
            "is_synthetic",
        ],
    )
    write_csv(
        output_dir / "figure9_power_accuracy.csv",
        figure9_rows,
        [
            "workload",
            "split",
            "signoff_total_mW",
            "capstone_prediction_mW",
            "oracle_prediction_mW",
            "capstone_ape_percent",
            "oracle_ape_percent",
            "is_synthetic",
        ],
    )

    capstone_metrics = metrics(
        [row["signoff_total_mW"] for row in figure9_rows],
        [row["capstone_prediction_mW"] for row in figure9_rows],
    )
    oracle_metrics = metrics(
        [row["signoff_total_mW"] for row in figure9_rows],
        [row["oracle_prediction_mW"] for row in figure9_rows],
    )
    train_rows = [row for row in figure9_rows if row["split"] == "train"]
    heldout_rows = [row for row in figure9_rows if row["split"] == "heldout"]

    metric_object = {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": is_synthetic,
        "warning": (
            "Synthetic functional demonstration; values do not reproduce the paper."
            if is_synthetic
            else "Private evaluation; do not commit without NDA clearance."
        ),
        "figure9_all": {
            "capstone": capstone_metrics,
            "oracle": oracle_metrics,
        },
        "figure9_train": {
            "capstone": metrics(
                [row["signoff_total_mW"] for row in train_rows],
                [row["capstone_prediction_mW"] for row in train_rows],
            )
        },
        "figure9_heldout": {
            "count": len(heldout_rows),
            "capstone": (
                metrics(
                    [row["signoff_total_mW"] for row in heldout_rows],
                    [row["capstone_prediction_mW"] for row in heldout_rows],
                )
                if heldout_rows
                else None
            ),
        },
        "figure10": {
            "workload": figure10_workload,
            "ptpx_breakdown_total_mW": ptpx_breakdown_total,
            "model_breakdown_total_mW": model_breakdown_total,
            "model_prediction_mW": model_prediction,
        },
    }
    write_json(output_dir / "power_model_metrics.json", metric_object)
    capstone_model_json = model_to_json(
        primary_model, is_synthetic=is_synthetic
    )
    capstone_model_json.update(
        {
            "model_role": "deployable_activity_proxy_model",
            "deployable": True,
            "fit_scope": "training_samples_only",
        }
    )
    write_json(
        output_dir / "capstone_power_model.json",
        capstone_model_json,
    )

    oracle_model_json = model_to_json(
        oracle_model, is_synthetic=is_synthetic
    )
    oracle_model_json.update(
        {
            "model_role": "evaluation_only_oracle",
            "deployable": False,
            "uses_sample_specific_fitted_gamma": True,
            "fit_scope": "all_samples_including_heldout",
        }
    )
    write_json(
        output_dir / "oracle_power_model.json",
        oracle_model_json,
    )
    write_json(
        output_dir / "run_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "is_synthetic": is_synthetic,
            "manifest": manifest_path.name,
            "power_unit_input": manifest["power_unit"],
            "power_unit_output": "mW",
            "training_names": primary_model.training_names,
            "heldout_names": [
                sample.name for sample in samples if sample.split == "heldout"
            ],
            "reference_calibration_workload": reference_name,
            "figure5_vecadd_only_scale_factor": vecadd_only_scale,
            "figure10_workload": figure10_workload,
            "dynamic_ridge": args.dynamic_ridge,
            "leakage_ridge": args.leakage_ridge,
            "gamma_iterations": args.gamma_iterations,
            "gamma_prior_strength": args.gamma_prior_strength,
        },
    )
    make_plots(
        figure5_rows,
        figure9_rows,
        figure10_mapping,
        figure10_breakdown,
        output_dir,
        is_synthetic,
    )

    if args.validate_reference:
        if not is_synthetic:
            raise ValueError(
                "--validate-reference is only defined for the bundled "
                "synthetic demonstration."
            )

        failures = []

        def check_close(
            label: str,
            actual: float,
            expected: float,
            tolerance: float = 1e-6,
        ) -> None:
            if not math.isclose(
                actual, expected, rel_tol=1e-9, abs_tol=tolerance
            ):
                failures.append(
                    f"{label}: expected {expected:.12g}, "
                    f"found {actual:.12g}"
                )

        if len(samples) != 11:
            failures.append(
                f"sample count: expected 11, found {len(samples)}"
            )
        if len(training_samples) != 8:
            failures.append(
                f"training count: expected 8, found {len(training_samples)}"
            )
        if len(heldout_rows) != 3:
            failures.append(
                f"held-out count: expected 3, found {len(heldout_rows)}"
            )
        check_close(
            "Figure 9 Capstone MAPE",
            float(capstone_metrics["MAPE_percent"]),
            7.543061458337864,
        )
        check_close(
            "Figure 9 oracle MAPE",
            float(oracle_metrics["MAPE_percent"]),
            0.8522108009301905,
        )
        check_close(
            "Figure 10 synthetic PTPX total",
            ptpx_breakdown_total,
            67.3546155592,
        )
        check_close(
            "Figure 10 model total",
            model_breakdown_total,
            67.3987825089986,
        )

        expected_outputs = [
            "capstone_power_model.json",
            "figure10_breakdown.csv",
            "figure10_mapping.csv",
            "figure10_mapping_and_breakdown.pdf",
            "figure10_mapping_and_breakdown.png",
            "figure5_fitting_scope.csv",
            "figure5_fitting_scope.pdf",
            "figure5_fitting_scope.png",
            "figure9_power_accuracy.csv",
            "figure9_power_accuracy.pdf",
            "figure9_power_accuracy.png",
            "oracle_power_model.json",
            "power_model_metrics.json",
            "run_provenance.json",
        ]
        missing_outputs = [
            filename
            for filename in expected_outputs
            if not (output_dir / filename).is_file()
        ]
        if missing_outputs:
            failures.append(
                "missing generated outputs: " + ", ".join(missing_outputs)
            )

        if failures:
            raise ValueError(
                "Reference validation failed:\n  - "
                + "\n  - ".join(failures)
            )
        print(
            "REFERENCE VALIDATION: PASS — bundled synthetic power-model "
            "data matches the reference signature."
        )

    print(f"Read {len(samples)} workloads from {manifest_path}")
    print(
        f"Primary model: {len(training_samples)} train and "
        f"{len(samples) - len(training_samples)} held-out workloads"
    )
    print(
        f"Figure 9 Capstone MAPE: {capstone_metrics['MAPE_percent']:.2f}%"
    )
    print(f"Figure 9 oracle MAPE: {oracle_metrics['MAPE_percent']:.2f}%")
    print(
        f"Figure 10 {figure10_workload}: PTPX={ptpx_breakdown_total:.3f} mW, "
        f"model={model_breakdown_total:.3f} mW"
    )
    print(
        f"Wrote Figure 5/9/10 data, plots, models, and metrics to {output_dir}"
    )
    if is_synthetic:
        print(
            "Synthetic demonstration only."
        )


def synthetic_feature_sets(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    names_and_scales = [
        ("vec_elemadd", 0.55, "train"),
        ("tensor3_ttv", 0.82, "train"),
        ("mat_elemmul", 0.95, "train"),
        ("mat_mask_tri", 1.32, "train"),
        ("mat_sddmm", 1.55, "train"),
        ("mat_mattransmul", 1.38, "train"),
        ("tensor3_mttkrp", 1.58, "train"),
        ("tensor3_innerprod", 1.08, "train"),
        ("harris", 1.18, "heldout"),
        ("unsharp", 1.45, "heldout"),
        ("gaussian", 0.72, "heldout"),
    ]
    rows = []
    for name, scale, split in names_and_scales:
        jitter = lambda magnitude: float(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
        features = {
            "num_pe_tiles": int(round(20 * scale * jitter(0.16))) + 3,
            "num_pe_ports": int(round(31 * scale * jitter(0.14))) + 4,
            "num_mem_tiles": int(round(8 * scale * jitter(0.15))) + 2,
            "num_mem_ports": int(round(13 * scale * jitter(0.15))) + 2,
            "num_io_tiles": int(round(7 * scale * jitter(0.10))) + 2,
            "num_ic_rmux": int(round(480 * scale * jitter(0.12))) + 25,
            "num_ic_reg": int(round(510 * scale * jitter(0.16))) + 20,
            "num_ic_port": int(round(76 * scale * jitter(0.12))) + 8,
            "num_ic_sb": int(round(970 * scale * jitter(0.10))) + 45,
            "num_pipeline_regs": int(round(250 * scale * jitter(0.18))) + 12,
        }
        rows.append({"name": name, "split": split, "features": features})
    return rows


def format_ptpx_row(
    indent: str,
    name: str,
    internal_w: float,
    switching_w: float,
    leakage_w: float,
    percent: float,
) -> str:
    total_w = internal_w + switching_w + leakage_w
    return (
        f"{indent}{name:<44} "
        f"{internal_w:.8e} {switching_w:.8e} {leakage_w:.8e} "
        f"{total_w:.8e} {percent:.2f}"
    )


def make_synthetic(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 101)

    # These are synthetic teaching parameters and are intentionally unrelated
    # to the coefficients used for the paper's real evaluation.
    teacher_dynamic = {
        "num_pe_tiles": 0.74,
        "num_pe_ports": 0.13,
        "num_mem_tiles": 1.08,
        "num_mem_ports": 0.16,
        "num_io_tiles": 0.48,
        "num_ic_rmux": 0.045,
        "num_ic_reg": 0.031,
        "num_ic_port": 0.038,
        "num_ic_sb": 0.024,
        "num_pipeline_regs": 0.026,
        "bias": 3.4,
    }
    teacher_leakage = {
        "num_pe_tiles": 0.010,
        "num_mem_tiles": 0.015,
        "num_io_tiles": 0.008,
        "num_pipeline_regs": 0.0012,
        "bias": 0.30,
    }

    datasets = synthetic_feature_sets(args.seed)
    manifest_datasets = []
    synthetic_truth = []
    for index, dataset in enumerate(datasets):
        features = dict(dataset["features"])
        features["bias"] = 1.0
        proxy = activity_proxy(features, DEFAULT_PROXY_WEIGHTS)
        # A deterministic workload component prevents a one-workload
        # calibration from generalizing perfectly, which is the behavior that
        # the Figure 5 demonstration is intended to expose.
        workload_component = 0.88 + 0.035 * index + float(rng.normal(0.0, 0.025))
        gamma = workload_component * (proxy / 1000.0) ** 0.09

        group_dynamic = {group: 0.0 for group in ROW_GROUPS}
        for feature, beta in teacher_dynamic.items():
            group = FEATURE_GROUP.get(feature, "OTHER")
            contribution = beta * float(features.get(feature, 0.0)) * gamma
            group_dynamic[group] += contribution

        # Small independent group noise makes the fitting task realistic while
        # preserving nonnegative, internally consistent synthetic reports.
        for group in ROW_GROUPS:
            if group_dynamic[group] > 0:
                group_dynamic[group] *= float(rng.uniform(0.975, 1.025))

        leakage_total = sum(
            teacher_leakage[feature] * float(features.get(feature, 0.0))
            for feature in teacher_leakage
        )
        leakage_total *= float(rng.uniform(0.985, 1.015))
        dynamic_total = sum(group_dynamic.values())
        total_mw = dynamic_total + leakage_total

        group_names = {
            "PE": "pe_cluster (synthetic_pe_tile)",
            "MEM": "mem_cluster (synthetic_mem_tile)",
            "IO": "io_cluster (synthetic_io_tile)",
            "SB": "switchbox (synthetic_SB_route)",
            "RMUX": "rmux (synthetic_RMUX_route)",
            "REG": "route_reg (synthetic_REG_route)",
            "PIPE": "pipeline (synthetic_pipeline_reg)",
            "PORT": "port (synthetic_PORT_route)",
            "OTHER": "other_logic",
            "SELF": "self_logic",
        }
        positive_groups = [
            group for group in ROW_GROUPS if group_dynamic[group] > 0
        ]
        lines = [
            "Synthetic PrimeTime PX-like hierarchy report",
            "Hierarchy                                      Int Power Switch Power Leak Power Total Power %",
            "-----------------------------------------------------------------------------------------------",
        ]
        root_internal = 0.72 * dynamic_total / 1000.0
        root_switching = 0.28 * dynamic_total / 1000.0
        lines.append(
            format_ptpx_row(
                "",
                "BLOCK_0_application_graph",
                root_internal,
                root_switching,
                leakage_total / 1000.0,
                100.0,
            )
        )
        for group in positive_groups:
            dynamic_mw = group_dynamic[group]
            fraction = dynamic_mw / max(dynamic_total, EPS)
            leak_mw = leakage_total * fraction
            lines.append(
                format_ptpx_row(
                    "  ",
                    group_names[group],
                    0.72 * dynamic_mw / 1000.0,
                    0.28 * dynamic_mw / 1000.0,
                    leak_mw / 1000.0,
                    100.0 * (dynamic_mw + leak_mw) / max(total_mw, EPS),
                )
            )
        lines.append("")

        report_name = f"{dataset['name']}.txt"
        (reports_dir / report_name).write_text(
            "\n".join(lines), encoding="utf-8"
        )
        manifest_datasets.append(
            {
                "name": dataset["name"],
                "split": dataset["split"],
                "report": f"reports/{report_name}",
                "features": dataset["features"],
            }
        )
        synthetic_truth.append(
            {
                "workload": dataset["name"],
                "split": dataset["split"],
                "gamma": gamma,
                "dynamic_mW": dynamic_total,
                "leakage_mW": leakage_total,
                "total_mW": total_mw,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": True,
        "description": (
            "Deterministic synthetic PTPX-like data for demonstrating the "
            "Capstone power model workflow. Not derived from the NDA-protected reports."
        ),
        "seed": args.seed,
        "power_unit": "W",
        "root_candidates": ["BLOCK_0_application_graph", "application_graph"],
        "hierarchy_mode": "direct_children",
        "consistency_tolerance": 1e-4,
        "reference_calibration_workload": "vec_elemadd",
        "figure10_workload": "vec_elemadd",
        "feature_order": FEATURE_ORDER,
        "leak_feature_order": LEAK_FEATURE_ORDER,
        "proxy_weights": DEFAULT_PROXY_WEIGHTS,
        "datasets": manifest_datasets,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "synthetic_generation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "is_synthetic": True,
            "seed": args.seed,
            "teacher_dynamic_mw_per_count": teacher_dynamic,
            "teacher_leakage_mw_per_count": teacher_leakage,
            "per_workload_truth": synthetic_truth,
            "warning": (
                "Synthetic values are not derived from and do not reproduce "
                "the paper's NDA-protected PTPX data."
            ),
        },
    )
    print(f"Wrote synthetic manifest and {len(datasets)} reports to {output_dir}")
    print(
        "Next: python3 src/capstone_power_model.py evaluate "
        f"--manifest {output_dir / 'manifest.json'} "
        "--output-dir generated_figures/generated_power_model"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser(
        "make-synthetic",
        help="Generate deterministic synthetic PTPX-like reports and a manifest.",
    )
    synthetic_parser.add_argument("--output-dir", type=Path, required=True)
    synthetic_parser.add_argument("--seed", type=int, default=7)
    synthetic_parser.add_argument("--overwrite", action="store_true")
    synthetic_parser.set_defaults(func=make_synthetic)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Fit the model and generate synthetic Figure 5/9/10 reproductions.",
    )
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--dynamic-ridge", type=float, default=1e-4)
    evaluate_parser.add_argument("--leakage-ridge", type=float, default=1e-6)
    evaluate_parser.add_argument("--gamma-iterations", type=int, default=12)
    evaluate_parser.add_argument(
        "--gamma-prior-strength",
        type=float,
        default=0.10,
        help="Shrink fitted activity factors toward one.",
    )
    evaluate_parser.add_argument(
        "--allow-private-inputs",
        action="store_true",
        help=(
            "Explicitly permit a manifest with is_synthetic=false. Keep all "
            "reports and generated values outside the public repository."
        ),
    )
    evaluate_parser.add_argument("--overwrite", action="store_true")
    evaluate_parser.add_argument(
        "--validate-reference",
        action="store_true",
        help=(
            "Validate the bundled synthetic data and expected outputs "
            "against the artifact's numerical reference signature."
        ),
    )
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "dynamic_ridge", 0.0) < 0:
        parser.error("--dynamic-ridge must be nonnegative.")
    if getattr(args, "leakage_ridge", 0.0) < 0:
        parser.error("--leakage-ridge must be nonnegative.")
    if getattr(args, "gamma_iterations", 1) <= 0:
        parser.error("--gamma-iterations must be positive.")
    if getattr(args, "gamma_prior_strength", 0.0) < 0:
        parser.error("--gamma-prior-strength must be nonnegative.")
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
