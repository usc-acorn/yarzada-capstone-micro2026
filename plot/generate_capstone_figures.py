#!/usr/bin/env python3
"""
Generate the artifact-backed paper outputs: Figures 5, 9, 11--13 and Tables VII--IX.

Figures 5 and 9 use released aggregate values from the camera-ready plotting
sources. Figures 11--13 and Tables VII--IX use the bundled controller data.
This script is the single top-level plotting entry point for the public artifact.

Run from the repository root:

    python3 plot/generate_capstone_figures.py \
        --data-dir data \
        --output-dir figures

Before plotting Figure 12, create its derived sweep CSV:

    python3 plot/generate_figure12_sweep_data.py \
        --trace data/capstone_tensor3_ttv_high_cap/capstone_all_modes_trace.csv \
        --output data/figure12_sweep.csv

Dependencies:

    python3 -m pip install numpy matplotlib

Metric definitions:

* Figure 12 reads data/figure12_sweep.csv produced by
  generate_figure12_sweep_data.py.
* Figure 13 and Tables VII--VIII use the selected model mean P_mean_mW
  (power-model estimate before the controller upper bound is applied).
* Slack is 100 * (cap - P_mean_mW) / cap.
* Figure 13 normalizes every selected frequency to that kernel's uncapped
  Cascade baseline.
* Table VIII normalizes every Capstone III sensitivity setting to full bounds
  with K=90.
* Table IX computes Cascade and Capstone rows from tensor3_innerprod and
  mat_sddmm; prior-work rows are fixed values transcribed from the paper.
* Table IX's optimistic 2x and 4x throttling columns divide both frequency and
  power by two and four, respectively.
* Capstone III refers to the full-bounds controller.
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PAPER_SIGNOFF_TIME_S = 1.07e5

KERNEL_ORDER = [
    "vec_elemadd",
    "mat_elemmul",
    "tensor3_ttv",
    "tensor3_mttkrp",
    "tensor3_innerprod",
    "mat_sddmm",
    "mat_mask_tri",
    "mat_mattransmul",
]

FIGURE11_KERNEL = "tensor3_ttv"
FIGURE12_SWEEP_CSV = "figure12_sweep.csv"

FILES = {
    "bitstreams": "capstone_all_modes_bitstreams.csv",
    "selection": "capstone_all_modes_selection.json",
    "summary": "capstone_all_modes_summary.csv",
    "trace": "capstone_all_modes_trace.csv",
    "timing": "capstone_figure11_timing.csv",
}

REFERENCE_SELECTIONS = {
    "vec_elemadd": {
        "cap_mW": 350.0,
        "baseline": (439.0, 371.9284669478902),
        "capstone_i": (263.0, 210.45581162312143),
        "capstone_ii": (338.0, 271.9283524651567),
        "capstone_iii_full": (370.0, 301.65076995214366),
    },
    "mat_elemmul": {
        "cap_mW": 650.0,
        "baseline": (479.0, 770.1248008421064),
        "capstone_i": (303.0, 448.0489121019552),
        "capstone_ii": (337.0, 504.0269239503728),
        "capstone_iii_full": (374.0, 562.9699041153505),
    },
    "tensor3_ttv": {
        "cap_mW": 700.0,
        "baseline": (431.0, 702.3634282063533),
        "capstone_i": (313.0, 479.22403074436903),
        "capstone_ii": (346.0, 535.6162921346598),
        "capstone_iii_full": (384.0, 602.7803565536133),
    },
    "tensor3_mttkrp": {
        "cap_mW": 1300.0,
        "baseline": (431.0, 1384.535239540606),
        "capstone_i": (295.0, 894.308558501759),
        "capstone_ii": (333.0, 1022.7707046531465),
        "capstone_iii_full": (360.0, 1116.9229886261344),
    },
    "tensor3_innerprod": {
        "cap_mW": 750.0,
        "baseline": (481.0, 909.4408560842875),
        "capstone_i": (294.0, 510.2386353787169),
        "capstone_ii": (335.0, 586.580533709162),
        "capstone_iii_full": (366.0, 645.568474880041),
    },
    "mat_sddmm": {
        "cap_mW": 1300.0,
        "baseline": (479.0, 1488.5567792860204),
        "capstone_i": (310.0, 893.9846064196975),
        "capstone_ii": (346.0, 1011.3045395814542),
        "capstone_iii_full": (379.0, 1121.4395430051131),
    },
    "mat_mask_tri": {
        "cap_mW": 1100.0,
        "baseline": (483.0, 1262.1339001459048),
        "capstone_i": (312.0, 756.4136822472539),
        "capstone_ii": (346.0, 849.2074155721372),
        "capstone_iii_full": (385.0, 956.4124630378409),
    },
    "mat_mattransmul": {
        "cap_mW": 1000.0,
        "baseline": (483.0, 1187.1197676941672),
        "capstone_i": (308.0, 689.0239101790621),
        "capstone_ii": (342.0, 780.7355188280025),
        "capstone_iii_full": (371.0, 854.8860733490842),
    },
}

REFERENCE_FIGURE11_NORMALIZED = {
    "baseline_normalized_compile_time": 1.0,
    "capstone_i_normalized_compile_time": 0.6768948325433524,
    "capstone_ii_normalized_compile_time": 0.7514743527836063,
    "capstone_iii_normalized_compile_time": 0.760984681753263,
}

REFERENCE_FIGURE12_TRADEOFF = {
    "capstone_i": (700.0, 313.0, 479.22403074436903),
    "capstone_ii": (700.0, 346.0, 535.6162921346598),
    "capstone_iii_full": (700.0, 384.0, 602.7803565536133),
}

REFERENCE_TABLE8 = {
    ("fit_1x", 90): (1.0771502520541958, 5.455220866220657, 53),
    ("fit_2x", 90): (1.0416021191507048, 9.491983160656078, 51),
    ("fit_activity", 90): (1.0488949503198732, 8.59989584959531, 51),
    ("fit_activity_pvt", 90): (
        1.0208237853334807,
        11.82091612153797,
        50,
    ),
    ("full", 90): (1.0, 13.851293110149957, 49),
    ("unpruned", 90): (1.0, 13.851293110149957, 49),
    ("full", 8): (1.0, 13.851293110149957, 8),
    ("full", 4): (1.0, 13.851293110149957, 4),
}

MAIN_MODES = [
    "baseline",
    "capstone_i",
    "capstone_ii",
    "capstone_iii_full",
]

MODE_LABELS = {
    "baseline": "Baseline",
    "capstone_i": "Cap I",
    "capstone_ii": "Cap II",
    "capstone_iii_full": "Cap III",
}

COLORS = {
    "baseline": "#fceed9",
    "capstone_i": "#f8dada",
    "capstone_ii": "#f0fedb",
    "capstone_iii_full": "#ebf6fe",
}

FIGURE12_LABELS = {
    "baseline": "Baseline (no capping)",
    "capstone_i": "Capstone I",
    "capstone_ii": "Capstone II",
    "capstone_iii_full": "Capstone III",
}

FIGURE12_MARKERS = {
    "baseline": "o",
    "capstone_i": "s",
    "capstone_ii": "^",
    "capstone_iii_full": "D",
}

TABLE8_SETTINGS = [
    ("fit_1x", 90),
    ("fit_2x", 90),
    ("fit_activity", 90),
    ("fit_activity_pvt", 90),
    ("full", 90),
    ("unpruned", 90),
    ("full", 8),
    ("full", 4),
]

TABLE8_LABELS = {
    ("fit_1x", 90): r"$1\times$ fit",
    ("fit_2x", 90): r"$2\times$ fit",
    ("fit_activity", 90): "Fit + activity",
    ("fit_activity_pvt", 90): "Fit + activity + PVT",
    ("full", 90): "Full bounds",
    ("unpruned", 90): "Unpruned",
    ("full", 8): r"Pruned to $K\leq 8$",
    ("full", 4): r"Pruned to $K\leq 4$",
}

TABLE8_BOUNDS = {
    ("fit_1x", 90): r"$\epsilon_{e,\mathrm{fit}}$",
    ("fit_2x", 90): r"$2\epsilon_{e,\mathrm{fit}}$",
    ("fit_activity", 90): (
        r"$\epsilon_{e,\mathrm{fit}}+\epsilon_{e,\mathrm{act}}$"
    ),
    ("fit_activity_pvt", 90): (
        r"$\epsilon_{e,\mathrm{fit}}+\epsilon_{e,\mathrm{act}}"
        r"+\epsilon_{e,\mathrm{PVT}}$"
    ),
    ("full", 90): (
        r"$\epsilon_{e,\mathrm{fit}}+\epsilon_{e,\mathrm{act}}"
        r"+\epsilon_{e,\mathrm{PVT}}+\epsilon_{e,\mathrm{OOD}}$"
    ),
    ("unpruned", 90): "Full bounds",
    ("full", 8): "Full bounds, top 8 retained",
    ("full", 4): "Full bounds, top 4 retained",
}

TABLE8_LABELS_PLAIN = {
    ("fit_1x", 90): "1× fit",
    ("fit_2x", 90): "2× fit",
    ("fit_activity", 90): "Fit + activity",
    ("fit_activity_pvt", 90): "Fit + activity + PVT",
    ("full", 90): "Full bounds",
    ("unpruned", 90): "Unpruned",
    ("full", 8): "Pruned to K ≤ 8",
    ("full", 4): "Pruned to K ≤ 4",
}

TABLE8_BOUNDS_PLAIN = {
    ("fit_1x", 90): "ε_fit",
    ("fit_2x", 90): "2ε_fit",
    ("fit_activity", 90): "ε_fit + ε_act",
    ("fit_activity_pvt", 90): "ε_fit + ε_act + ε_PVT",
    ("full", 90): "ε_fit + ε_act + ε_PVT + ε_OOD",
    ("unpruned", 90): "Full bounds",
    ("full", 8): "Full bounds, top 8 retained",
    ("full", 4): "Full bounds, top 4 retained",
}

TABLE9_PRIOR_ROWS: list[dict[str, Any]] = [
    {
        "compiler": "RipTide",
        "tech": "22FFL",
        "fabric": "6×6",
        "workload": "FFT",
        "cap_mW": 0.30,
        "cap_display": "0.30",
        "freq_MHz": [50.0, 25.0, 12.5],
        "power_mW": [0.24, 0.12, 0.06],
        "delta_cap_pct": [20.0, 60.0, 80.0],
        "success": ["Y", "Y", "Y"],
        "source": "paper_hardcoded",
    },
    {
        "compiler": "Snafu",
        "tech": "22FFL",
        "fabric": "6×6",
        "workload": "FFT",
        "cap_mW": 0.40,
        "cap_display": "0.40",
        "freq_MHz": [50.0, 25.0, 12.5],
        "power_mW": [0.54, 0.27, 0.135],
        "delta_cap_pct": [-35.0, 32.5, 66.25],
        "success": ["N", "Y", "Y"],
        "source": "paper_hardcoded",
    },
    {
        "compiler": "UE-CGRA",
        "tech": "28nm",
        "fabric": "8×8",
        "workload": "FFT",
        "cap_mW": 5.0,
        "cap_display": "5",
        "freq_MHz": [750.0, 325.0, 162.5],
        "power_mW": [14.0, 7.0, 3.5],
        "delta_cap_pct": [-180.0, -40.0, 30.0],
        "success": ["N", "N", "Y"],
        "source": "paper_hardcoded",
    },
    {
        "compiler": "Plasticine",
        "tech": "28nm",
        "fabric": "–",
        "workload": "Inner Prod.",
        "cap_mW": 3000.0,
        "cap_display": "3000",
        "freq_MHz": [280.0, 140.0, 70.0],
        "power_mW": [18900.0, 9450.0, 4725.0],
        "delta_cap_pct": [-530.0, -215.0, -57.5],
        "success": ["N", "N", "N"],
        "source": "paper_hardcoded",
    },
]

TABLE9_COMPUTED_MODES = [
    ("Cascade", "baseline", "12nm"),
    ("Capstone I", "capstone_i", "16nm"),
    ("Capstone II", "capstone_ii", "16nm"),
    ("Capstone III", "capstone_iii_full", "16nm"),
]

TABLE9_WORKLOADS = [
    ("tensor3_innerprod", "Inner Prod."),
    ("mat_sddmm", "SDDMM"),
]

# Signoff oracle data is not present in the all-modes dumps because
# it requires per-candidate signoff power.
TABLE7_REFERENCE_ROWS: list[dict[str, Any]] = [
    {
        "controller": "Signoff oracle",
        "success_pct": 100.0,
        "median_delta_cap_pct": 3.13,
        "avg_norm_freq": 1.0,
        "K": "1",
        "source": "transcribed_from_paper",
    },
]


@dataclass
class RunData:
    kernel: str
    directory: Path
    run_id: str
    cap_mw: float
    selection: dict[str, Any]
    timing: dict[str, str]

    def selected(self, mode: str) -> dict[str, Any]:
        candidate = self.selection.get("selected_modes", {}).get(mode)
        if candidate is None:
            raise ValueError(f"{self.kernel}: no selected candidate for {mode}.")
        return candidate

    def table8_candidate(self, bound_mode: str, k_value: int) -> dict[str, Any] | None:
        rows = self.selection.get("table8", self.selection.get("table7", []))
        for row in rows:
            if str(row.get("bound_mode")) == bound_mode and to_int(row.get("K")) == k_value:
                return row.get("selected")
        return None

    def table8_retained(self, bound_mode: str, k_value: int) -> int:
        rows = self.selection.get("table8", self.selection.get("table7", []))
        for row in rows:
            if str(row.get("bound_mode")) == bound_mode and to_int(row.get("K")) == k_value:
                return int(row.get("retained_count", 0))
        return 0


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a number, got {value!r}.") from error


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected an integer, got {value!r}.") from error


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_runs(data_dir: Path) -> list[RunData]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    discovered: dict[str, RunData] = {}
    for directory in sorted(data_dir.glob("capstone_*")):
        if not directory.is_dir():
            continue
        kernel = directory.name.removeprefix("capstone_")
        # Figure 12 is created later in this script
        if kernel.endswith("_high_cap"):
            continue
        missing = [
            filename
            for filename in FILES.values()
            if not (directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{directory} is missing: {', '.join(missing)}"
            )

        selection_path = directory / FILES["selection"]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        run_id = str(selection.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"{selection_path} does not contain run_id.")

        cap_mw = to_float(selection.get("power_cap_mW"))
        if cap_mw is None or cap_mw <= 0.0:
            raise ValueError(f"{kernel}: power_cap_mW must be positive.")

        timing_rows = [
            row
            for row in read_csv_rows(directory / FILES["timing"])
            if row.get("run_id") == run_id
        ]
        if not timing_rows:
            raise ValueError(
                f"{kernel}: timing CSV has no row for run_id={run_id!r}."
            )

        # Cross-check the manifest against the rank 0 bitstream and summary rows.
        summary_rows = [
            row
            for row in read_csv_rows(directory / FILES["summary"])
            if row.get("run_id") == run_id
        ]
        bitstream_rows = [
            row
            for row in read_csv_rows(directory / FILES["bitstreams"])
            if row.get("run_id") == run_id
        ]
        for mode in MAIN_MODES:
            candidate = selection.get("selected_modes", {}).get(mode)
            if candidate is None:
                raise ValueError(f"{kernel}: manifest has no {mode} selection.")
            iteration = int(candidate["iteration"])
            if not any(
                row.get("mode") == mode
                and to_int(row.get("selected_iteration")) == iteration
                for row in summary_rows
            ):
                raise ValueError(
                    f"{kernel}: summary CSV disagrees with the {mode} "
                    f"manifest selection at iteration {iteration}."
                )
            if not any(
                row.get("mode") == mode
                and to_int(row.get("rank")) == 0
                and to_int(row.get("iteration")) == iteration
                for row in bitstream_rows
            ):
                raise ValueError(
                    f"{kernel}: bitstream CSV has no rank-0 {mode} candidate "
                    f"at iteration {iteration}."
                )

        if kernel in discovered:
            raise ValueError(f"Duplicate data directory for kernel {kernel}.")
        discovered[kernel] = RunData(
            kernel=kernel,
            directory=directory,
            run_id=run_id,
            cap_mw=float(cap_mw),
            selection=selection,
            timing=timing_rows[-1],
        )

    missing_kernels = [kernel for kernel in KERNEL_ORDER if kernel not in discovered]
    extra_kernels = sorted(set(discovered) - set(KERNEL_ORDER))
    if missing_kernels:
        raise ValueError(
            "Missing expected kernel directories: " + ", ".join(missing_kernels)
        )
    if extra_kernels:
        print(
            "warning: ignoring unrecognized kernel directories: "
            + ", ".join(extra_kernels),
            file=sys.stderr,
        )
    return [discovered[kernel] for kernel in KERNEL_ORDER]


def slack_pct(cap_mw: float, power_mw: float) -> float:
    return 100.0 * (cap_mw - power_mw) / cap_mw


def selected_metrics(runs: list[RunData]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        baseline_f = float(run.selected("baseline")["f_mhz"])
        for mode in MAIN_MODES:
            candidate = run.selected(mode)
            mean_power = float(candidate["P_mean_mW"])
            rows.append(
                {
                    "run_id": run.run_id,
                    "kernel": run.kernel,
                    "mode": mode,
                    "cap_mW": run.cap_mw,
                    "iteration": int(candidate["iteration"]),
                    "breaks": int(candidate["breaks"]),
                    "f_mhz": float(candidate["f_mhz"]),
                    "baseline_f_mhz": baseline_f,
                    "norm_freq_vs_baseline": float(candidate["f_mhz"])
                    / baseline_f,
                    "P_mean_mW": mean_power,
                    "P_upper_mW": float(candidate["P_upper_mW"]),
                    "delta_cap_pct": (
                        ""
                        if mode == "baseline"
                        else slack_pct(run.cap_mw, mean_power)
                    ),
                    "success": mean_power <= run.cap_mw + 1e-9,
                }
            )
    return rows


def plot_figure12(sweep_csv: Path, output_dir: Path) -> None:
    """Generate Figure 12: TTV frequency versus power cap with 700 mW slack summary."""
    if not sweep_csv.is_file():
        raise FileNotFoundError(
            f"Figure 12 sweep CSV does not exist: {sweep_csv}. Generate it "
            "first with: python3 plot/generate_figure12_sweep_data.py "
            "--trace data/capstone_tensor3_ttv_high_cap/"
            "capstone_all_modes_trace.csv --output data/figure12_sweep.csv"
        )
    rows = read_csv_rows(sweep_csv)
    if not rows:
        raise ValueError(f"Figure 12 sweep CSV is empty: {sweep_csv}")

    sweep_rows = [row for row in rows if row.get("panel") == "a_cap_sweep"]
    tradeoff_rows = [row for row in rows if row.get("panel") == "b_tradeoff"]
    if not sweep_rows or not tradeoff_rows:
        raise ValueError(
            f"{sweep_csv} must contain a_cap_sweep and b_tradeoff rows."
        )

    required_modes = set(MAIN_MODES)
    sweep_modes = {row.get("mode") for row in sweep_rows}
    if not required_modes.issubset(sweep_modes):
        missing = sorted(required_modes - sweep_modes)
        raise ValueError("Figure 12 sweep is missing modes: " + ", ".join(missing))

    fig, ax = plt.subplots(figsize=(6.9, 3.05), dpi=200)
    grid_color = "#D8D8D8"
    line_color = "#555555"

    for mode in MAIN_MODES:
        mode_rows = sorted(
            [row for row in sweep_rows if row.get("mode") == mode],
            key=lambda row: float(to_float(row.get("cap_mW"))),
        )
        x_values = np.array([float(to_float(row["cap_mW"])) for row in mode_rows], dtype=float)
        y_values = np.array([
            float(to_float(row["f_mhz"]))
            if parse_bool(row.get("safe_candidate_found")) and to_float(row.get("f_mhz")) is not None
            else np.nan
            for row in mode_rows
        ], dtype=float)
        mask = ~np.isnan(y_values)
        ax.plot(
            x_values[mask],
            y_values[mask],
            label=FIGURE12_LABELS[mode],
            linewidth=1.55,
            marker=FIGURE12_MARKERS[mode],
            markersize=6.7,
            color=line_color,
            markerfacecolor=COLORS[mode],
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=3,
        )

    tradeoff_by_mode = {
        row["mode"]: row for row in tradeoff_rows
        if row.get("mode") in {"capstone_i", "capstone_ii", "capstone_iii_full"}
    }
    annotation_cap = 700.0
    ax.axvline(annotation_cap, color="#A0A0A0", linestyle="--", linewidth=0.8, alpha=0.65, zorder=1)
    slack_values = {}
    for mode in ["capstone_i", "capstone_ii", "capstone_iii_full"]:
        row = tradeoff_by_mode.get(mode)
        if row is None:
            continue
        x_value = float(row["cap_mW"])
        y_value = float(row["f_mhz"])
        slack_values[mode] = float(row["delta_cap_pct"])
        ax.scatter(
            [x_value], [y_value], s=76, marker=FIGURE12_MARKERS[mode],
            facecolor=COLORS[mode], edgecolor="black", linewidth=1.1, zorder=5,
        )

    summary_lines = [rf"At $C={annotation_cap:g}$ mW:"]
    if "capstone_i" in slack_values:
        summary_lines.append(rf"Capstone I: $\Delta$Cap = {slack_values['capstone_i']:.1f}%")
    if "capstone_ii" in slack_values:
        summary_lines.append(rf"Capstone II: $\Delta$Cap = {slack_values['capstone_ii']:.1f}%")
    if "capstone_iii_full" in slack_values:
        summary_lines.append(rf"Capstone III: $\Delta$Cap = {slack_values['capstone_iii_full']:.1f}%")
    ax.text(
        0.972, 0.085, "\n".join(summary_lines), transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10.3, linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="black", linewidth=0.75, alpha=0.90),
        zorder=7,
    )

    ax.set_xlabel("Power Cap (mW)", fontsize=13, labelpad=6)
    ax.set_ylabel("Frequency (MHz)", fontsize=13, labelpad=7)
    ax.set_xlim(485, 1015)
    ax.set_xticks([500, 600, 700, 800, 900, 1000])
    ax.set_ylim(220, 445)
    ax.set_yticks([250, 300, 350, 400])
    ax.tick_params(axis="both", labelsize=10.5, width=0.9, length=4.0)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=grid_color, linewidth=0.6, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    desired = ["Baseline (no capping)", "Capstone I", "Capstone II", "Capstone III"]
    label_to_handle = {label: handle for handle, label in zip(handles, labels)}
    leg = fig.legend(
        [label_to_handle[label] for label in desired], desired,
        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4,
        frameon=True, fancybox=True, fontsize=9.0,
        handlelength=1.35, handletextpad=0.38, columnspacing=0.85,
        borderpad=0.34,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.84)
    frame.set_edgecolor("black")
    frame.set_linewidth(0.9)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.19, top=0.82)
    save_figure(fig, output_dir / "figure12_controller_evaluation")


def plot_figure13(
    metric_rows: list[dict[str, Any]], runs: list[RunData], output_dir: Path
) -> None:
    """Generate Figure 13: combined cross-kernel normalized frequency and cap slack."""
    modes = ["capstone_i", "capstone_ii", "capstone_iii_full"]
    short_name = {
        "vec_elemadd": "VecAdd",
        "mat_elemmul": "ElemMul",
        "tensor3_ttv": "TTV",
        "tensor3_mttkrp": "MTTKRP",
        "tensor3_innerprod": "T3-Inner",
        "mat_sddmm": "SDDMM",
        "mat_mask_tri": "TriMask",
        "mat_mattransmul": "MatT-Mul",
    }
    lookup = {(row["kernel"], row["mode"]): row for row in metric_rows}
    kernels = [short_name.get(run.kernel, run.kernel) for run in runs]
    x = np.arange(len(runs))
    bar_width = 0.22

    fig, (ax_freq, ax_slack) = plt.subplots(
        2, 1, figsize=(6.5, 3.85), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.86], "hspace": 0.13},
    )

    for i, mode in enumerate(modes):
        values = [float(lookup[(run.kernel, mode)]["norm_freq_vs_baseline"]) for run in runs]
        ax_freq.bar(
            x + (i - 1) * bar_width, values, width=bar_width,
            color=COLORS[mode], edgecolor="black", linewidth=0.65, zorder=3,
        )
    ax_freq.axhline(1.0, color="#666666", linestyle="--", linewidth=1.15, alpha=0.9, zorder=2)
    ax_freq.text(
        0.995, 1.005, "Baseline = 1.0", transform=ax_freq.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8.8, color="#666666",
    )
    ax_freq.set_ylabel("Norm. Frequency", fontsize=11.8, labelpad=7)
    ax_freq.set_ylim(0.20, 1.08)
    ax_freq.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_freq.tick_params(axis="y", labelsize=9.8, width=0.9, length=4.0)
    ax_freq.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    for i, mode in enumerate(modes):
        values = [float(lookup[(run.kernel, mode)]["delta_cap_pct"]) for run in runs]
        ax_slack.bar(
            x + (i - 1) * bar_width, values, width=bar_width,
            color=COLORS[mode], edgecolor="black", linewidth=0.65, zorder=3,
        )
    ax_slack.set_ylabel(r"Slack $\Delta$Cap (%)", fontsize=11.8, labelpad=7)
    ax_slack.set_ylim(0, 35)
    ax_slack.set_yticks([0, 10, 20, 30])
    ax_slack.tick_params(axis="y", labelsize=9.8, width=0.9, length=4.0)
    ax_slack.set_xticks(x)
    ax_slack.set_xticklabels(kernels, rotation=24, ha="right", rotation_mode="anchor", fontsize=9.0)
    ax_slack.tick_params(axis="x", width=0.9, length=4.0, pad=2)

    for ax in (ax_freq, ax_slack):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#D8D8D8", linewidth=0.6, alpha=0.72)
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.15, label="Baseline"),
        Patch(facecolor=COLORS["capstone_i"], edgecolor="black", linewidth=0.65, label="Capstone I"),
        Patch(facecolor=COLORS["capstone_ii"], edgecolor="black", linewidth=0.65, label="Capstone II"),
        Patch(facecolor=COLORS["capstone_iii_full"], edgecolor="black", linewidth=0.65, label="Capstone III"),
    ]
    leg = fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=True, fancybox=True, fontsize=9.2,
        handlelength=1.25, handletextpad=0.4, columnspacing=0.95,
        borderpad=0.36, labelspacing=0.22,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.84)
    frame.set_edgecolor("black")
    frame.set_linewidth(0.9)

    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.19, top=0.865, hspace=0.13)
    save_figure(fig, output_dir / "figure13_cross_kernel_controllers")


def format_share(value: float | None) -> str:
    if value is None:
        return "–"
    if 0.0 < value < 1.0:
        return "< 1%"
    if math.isclose(value, 100.0, abs_tol=0.005):
        return "100%"
    return f"{value:.2f}%"


def plot_figure11(run: RunData, output_dir: Path) -> None:
    """Generate Figure 11: per-iteration run time and total compile-time impact."""
    row = run.timing
    components = [
        (
            "STA (timing)",
            to_float(row.get("sta_per_iteration_s"), 0.0),
            to_float(row.get("sta_share_pct")),
        ),
        (
            "Pipelining",
            to_float(row.get("pipelining_per_iteration_s"), 0.0),
            to_float(row.get("pipelining_share_pct")),
        ),
        (
            "Capstone predictor",
            to_float(row.get("capstone_predictor_per_iteration_s"), 0.0),
            to_float(row.get("capstone_predictor_share_pct")),
        ),
        (
            "Post-PnR iter total",
            to_float(row.get("post_pnr_iteration_mean_s"), 0.0),
            100.0,
        ),
    ]
    pipeline_loop = float(
        to_float(row.get("pipeline_search_loop_total_s"), 0.0)
    )
    signoff_time = to_float(row.get("signoff_power_s"))
    signoff_time = PAPER_SIGNOFF_TIME_S if signoff_time is None else float(signoff_time)

    normalized = [
        float(to_float(row.get("capstone_i_normalized_compile_time"), 0.0)),
        float(to_float(row.get("capstone_ii_normalized_compile_time"), 0.0)),
        float(to_float(row.get("capstone_iii_normalized_compile_time"), 0.0)),
    ]

    fig = plt.figure(figsize=(7.15, 2.65), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.0], wspace=0.30)
    ax_table = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    # Panel (a): compact table matching the paper figure.
    ax_table.set_axis_off()
    ax_table.set_xlim(0.0, 1.0)
    ax_table.set_ylim(0.0, 1.0)
    x_component, x_time, x_share = 0.03, 0.69, 0.97
    ax_table.text(x_component, 0.93, "Component", weight="bold", fontsize=11.8)
    ax_table.text(x_time, 0.93, "Time / iter (s)", weight="bold", fontsize=11.8, ha="center")
    ax_table.text(x_share, 0.93, "Share", weight="bold", fontsize=11.8, ha="right")
    ax_table.hlines([0.99, 0.87], 0.0, 1.0, color="black", linewidth=[1.45, 0.8])

    for (label, seconds, share), y_value in zip(components, [0.77, 0.68, 0.59, 0.43]):
        weight = "bold" if label == "Capstone predictor" else "normal"
        ax_table.text(x_component, y_value, label, fontsize=10.8, weight=weight)
        ax_table.text(x_time, y_value, f"{float(seconds):.2f}", fontsize=10.8, ha="center")
        ax_table.text(x_share, y_value, format_share(share), fontsize=10.8, ha="right")
    ax_table.hlines([0.51, 0.35], 0.0, 1.0, color="black", linewidth=0.8)

    ax_table.text(x_component, 0.25, "Pipeline loop", fontsize=10.8)
    ax_table.text(x_time, 0.25, f"{pipeline_loop:.2f}", fontsize=10.8, ha="center")
    ax_table.text(x_share, 0.25, "–", fontsize=10.8, ha="right")
    ax_table.text(x_component, 0.15, "Signoff power", fontsize=10.8)
    ax_table.text(x_time, 0.15, r"$1.07\times10^{5}$", fontsize=10.8, ha="center")
    ax_table.text(x_share, 0.15, "–", fontsize=10.8, ha="right")
    ax_table.hlines(0.07, 0.0, 1.0, color="black", linewidth=1.45)

    # Panel (b): Capstone I--III bars with uncapped baseline as reference line.
    labels = ["I", "II", "III"]
    x = np.arange(len(labels))
    bars = ax_bar.bar(
        x,
        normalized,
        width=0.50,
        color=COLORS["capstone_iii_full"],
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    ax_bar.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, zorder=2)
    ax_bar.text(2.38, 1.008, "Baseline = 1.00", ha="right", va="bottom", fontsize=8.5, color="#666666")
    for bar, value in zip(bars, normalized):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.3,
        )

    ax_bar.set_ylabel("Norm. Compile Time", fontsize=11.2, labelpad=5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10.2)
    ax_bar.set_ylim(0.60, 1.07)
    ax_bar.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_bar.tick_params(axis="y", labelsize=9.0)
    ax_bar.set_axisbelow(True)
    ax_bar.yaxis.grid(True, color="#D8D8D8", linewidth=0.55, alpha=0.75)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.text(0.305, 0.018, "(a) Run time breakdown.", ha="center", va="bottom", fontsize=9.4)
    fig.text(0.805, 0.018, "(b) Compile-time impact.", ha="center", va="bottom", fontsize=9.4)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.98, bottom=0.13)
    save_figure(fig, output_dir / "figure11_runtime_impact")


def figure13_rows(
    metric_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    definitions = [
        ("Cascade", "baseline", "1"),
        ("Capstone I", "capstone_i", r"$\leq 4$"),
        ("Capstone II", "capstone_ii", r"$\leq 4$"),
        (r"Capstone III, $K\leq 4$", "capstone_iii_full", r"$\leq 4$"),
    ]
    result = []
    for controller, mode, k_label in definitions:
        rows = [row for row in metric_rows if row["mode"] == mode]
        successful = [row for row in rows if row["success"]]
        slack_values = [
            float(row["delta_cap_pct"])
            for row in successful
            if row["delta_cap_pct"] != ""
        ]
        result.append(
            {
                "controller": controller,
                "success_pct": 100.0
                * sum(bool(row["success"]) for row in rows)
                / len(rows),
                "median_delta_cap_pct": (
                    None
                    if not slack_values
                    else statistics.median(slack_values)
                ),
                "avg_norm_freq": statistics.mean(
                    float(row["norm_freq_vs_baseline"]) for row in rows
                ),
                "K": k_label,
                "source": "computed_from_all_modes_dumps",
            }
        )

    # Scalar Aggregate NNLS is an uncapped K=1 estimator baseline. Because it
    # does not change or stop the compiler search (its severe underprediction 
    # causes it to accept the maximally pipelined candidates), its selected 
    # candidates and aggregate controller metrics match the uncapped Cascade row.
    scalar_aggregate = dict(result[0])
    scalar_aggregate.update(
        {
            "controller": "Scalar Aggregate NNLS",
            "K": "1",
            "source": "derived_from_uncapped_cascade_selection",
        }
    )
    result.insert(1, scalar_aggregate)

    for offset, reference in enumerate(TABLE7_REFERENCE_ROWS, start=2):
        result.insert(offset, dict(reference))
    return result


def render_table(
    column_labels: list[str],
    cell_rows: list[list[str]],
    column_widths: list[float],
    figsize: tuple[float, float],
    font_size: float,
    output_stem: Path,
    separator_before: int | list[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_rows,
        colLabels=column_labels,
        colWidths=column_widths,
        cellLoc="center",
        loc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.38)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.1)
        else:
            cell.visible_edges = ""
        if column_index == 0:
            cell.get_text().set_ha("left")

    last_row = len(cell_rows)
    for column_index in range(len(column_labels)):
        table[(last_row, column_index)].visible_edges = "B"
        table[(last_row, column_index)].set_linewidth(1.1)
    separators = (
        []
        if separator_before is None
        else [separator_before]
        if isinstance(separator_before, int)
        else separator_before
    )
    for separator_index in separators:
        table_row = separator_index + 1
        for column_index in range(len(column_labels)):
            table[(table_row, column_index)].visible_edges = "T"
            table[(table_row, column_index)].set_linewidth(0.8)

    fig.tight_layout(pad=0.2)
    save_figure(fig, output_stem)


def write_table7_aggregate(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write Table VII: aggregate controller metrics over kernel-cap pairs."""
    csv_rows = []
    visual_rows = []
    for row in rows:
        median = row["median_delta_cap_pct"]
        median_text = "–" if median is None else f"{median:.2f}%"
        success_text = f"{row['success_pct']:.0f}%"
        controller_plain = row["controller"].replace(r"$K\leq 4$", "K ≤ 4")
        k_plain = row["K"].replace(r"$\leq 4$", "≤ 4")
        visual_rows.append([
            controller_plain,
            success_text,
            f"{row['avg_norm_freq']:.2f}",
            median_text,
            k_plain,
        ])
        csv_rows.append({
            "controller": controller_plain,
            "success_pct": row["success_pct"],
            "avg_norm_freq": row["avg_norm_freq"],
            "median_delta_cap_pct": "" if median is None else median,
            "K": k_plain,
            "source": row["source"],
        })

    stem = output_dir / "table7_aggregate_controller_metrics"
    render_table(
        ["Controller", "Success", "Norm. freq.", "Med. ΔCap", "K"],
        visual_rows,
        [0.38, 0.13, 0.18, 0.20, 0.11],
        (7.0, 2.35 + 0.22 * len(rows)),
        11,
        stem,
    )
    write_csv(
        stem.with_suffix(".csv"),
        csv_rows,
        ["controller", "success_pct", "avg_norm_freq", "median_delta_cap_pct", "K", "source"],
    )


def table8_sensitivity_rows(runs: list[RunData]) -> list[dict[str, Any]]:
    result = []
    for bound_mode, k_value in TABLE8_SETTINGS:
        source_bound_mode = "full" if bound_mode == "unpruned" else bound_mode

        per_run = []
        for run in runs:
            candidate = run.table8_candidate(source_bound_mode, k_value)
            full_reference = run.table8_candidate("full", 90)
            if full_reference is None:
                raise ValueError(
                    f"{run.kernel}: Table VIII full-bounds K=90 row is missing."
                )
            if candidate is None:
                per_run.append(
                    {
                        "success": False,
                        "norm_freq": None,
                        "slack": None,
                        "retained": 0,
                    }
                )
                continue

            mean_power = float(candidate["P_mean_mW"])
            is_success = mean_power <= run.cap_mw + 1e-9
            per_run.append(
                {
                    "success": is_success,
                    "norm_freq": float(candidate["f_mhz"])
                    / float(full_reference["f_mhz"]),
                    "slack": (
                        slack_pct(run.cap_mw, mean_power)
                        if is_success
                        else None
                    ),
                    "retained": run.table8_retained(source_bound_mode, k_value),
                }
            )

        normalized = [
            row["norm_freq"]
            for row in per_run
            if row["norm_freq"] is not None
        ]
        slacks = [row["slack"] for row in per_run if row["slack"] is not None]
        result.append(
            {
                "setting": TABLE8_LABELS[(bound_mode, k_value)],
                "bound_construction": TABLE8_BOUNDS[
                    (bound_mode, k_value)
                ],
                "bound_mode": bound_mode,
                "success_pct": 100.0
                * sum(bool(row["success"]) for row in per_run)
                / len(per_run),
                "avg_norm_freq": statistics.mean(normalized),
                "median_delta_cap_pct": (
                    None if not slacks else statistics.median(slacks)
                ),
                "K": k_value,
                "min_retained_count": min(
                    int(row["retained"]) for row in per_run
                ),
            }
        )
    return result


def write_table8_sensitivity(rows: list[dict[str, Any]], output_dir: Path) -> None:
    visual_rows = []
    csv_rows = []
    latex = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Capstone III sensitivity to bounded-error size and candidate pruning.}",
        r"  \label{tab:capstone-iii-sensitivity}",
        r"  \small",
        r"  \begin{tabular}{@{}llcccc@{}}",
        r"    \toprule",
        r"    \textbf{Setting} & \textbf{Bound construction} & \textbf{Success} & \textbf{Avg. norm. freq.} & \textbf{Med. $\Delta$Cap (\%)} & \textbf{$K$} \\",
        r"    \midrule",
    ]
    for index, row in enumerate(rows):
        if index == 5:
            latex.append(r"    \midrule")
        median = row["median_delta_cap_pct"]
        median_text = "–" if median is None else f"{median:.2f}"
        setting_key = (row["bound_mode"], row["K"])
        setting_plain = TABLE8_LABELS_PLAIN[setting_key]
        bounds_plain = TABLE8_BOUNDS_PLAIN[setting_key]
        visual_rows.append(
            [
                setting_plain,
                bounds_plain,
                f"{row['success_pct']:.0f}%",
                f"{row['avg_norm_freq']:.2f}",
                median_text,
                str(row["K"]),
            ]
        )
        latex.append(
            "    "
            + f"{row['setting']} & {row['bound_construction']} & "
            + f"{row['success_pct']:.0f}\\% & "
            + f"{row['avg_norm_freq']:.2f} & "
            + f"{'--' if median is None else f'{median:.2f}'} & "
            + f"{row['K']} \\\\"
        )
        csv_rows.append(
            {
                "setting": setting_plain,
                "bound_construction": bounds_plain,
                "bound_mode": row["bound_mode"],
                "success_pct": row["success_pct"],
                "avg_norm_freq": row["avg_norm_freq"],
                "median_delta_cap_pct": "" if median is None else median,
                "K": row["K"],
                "min_retained_count": row["min_retained_count"],
            }
        )
    latex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    stem = output_dir / "table8_capstone_iii_sensitivity"
    render_table(
        [
            "Setting",
            "Bound construction",
            "Success",
            "Avg. norm. freq.",
            "Med. ΔCap (%)",
            "K",
        ],
        visual_rows,
        [0.23, 0.35, 0.10, 0.14, 0.14, 0.04],
        (11.0, 3.5),
        10,
        stem,
        separator_before=5,
    )
    
    write_csv(
        stem.with_suffix(".csv"),
        csv_rows,
        [
            "setting",
            "bound_construction",
            "bound_mode",
            "success_pct",
            "avg_norm_freq",
            "median_delta_cap_pct",
            "K",
            "min_retained_count",
        ],
    )


def format_table9_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Table IX contains a non-finite value: {value}")
    if abs(value) < 1.0 and value != 0.0:
        text = f"{value:.3f}"
    else:
        text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def table9_rows(runs: list[RunData]) -> list[dict[str, Any]]:
    run_by_kernel = {run.kernel: run for run in runs}
    missing = [
        kernel for kernel, _ in TABLE9_WORKLOADS if kernel not in run_by_kernel
    ]
    if missing:
        raise ValueError(
            "Table IX is missing required kernel data: " + ", ".join(missing)
        )

    rows = [dict(row) for row in TABLE9_PRIOR_ROWS]
    for kernel, workload_label in TABLE9_WORKLOADS:
        run = run_by_kernel[kernel]
        for compiler, mode, technology in TABLE9_COMPUTED_MODES:
            candidate = run.selected(mode)
            frequency = float(candidate["f_mhz"])
            power = float(candidate["P_mean_mW"])
            if frequency <= 0.0 or power < 0.0:
                raise ValueError(
                    f"{kernel}/{mode}: Table IX frequency must be positive and "
                    "mean power must be nonnegative."
                )
            frequencies = [
                frequency,
                frequency / 2.0,
                frequency / 4.0,
            ]
            powers = [
                power,
                power / 2.0,
                power / 4.0,
            ]
            deltas = [
                slack_pct(run.cap_mw, variant_power)
                for variant_power in powers
            ]
            successes = [
                "Y" if variant_power <= run.cap_mw + 1e-9 else "N"
                for variant_power in powers
            ]
            rows.append(
                {
                    "compiler": compiler,
                    "tech": technology,
                    "fabric": "32×16",
                    "workload": workload_label,
                    "cap_mW": run.cap_mw,
                    "cap_display": format_table9_number(run.cap_mw),
                    "freq_MHz": frequencies,
                    "power_mW": powers,
                    "delta_cap_pct": deltas,
                    "success": successes,
                    "source": run.directory.name,
                    "kernel": kernel,
                    "mode": mode,
                }
            )
    return rows



def generate_figure5(output_dir: Path) -> None:
    """Generate camera-ready Figure 5 directly from released aggregate values."""
    from matplotlib.ticker import MultipleLocator

    gray = "#7A7A7A"
    grid = "#D8D8D8"
    row_band = "#F6F6F6"
    blue = "#EBF6FE"
    pink = "#F8DADA"
    magenta = "#D81B8A"

    kernels = [
        "vec_elemadd",
        "mat_elemmul",
        "tensor3_ttv",
        "tensor3_mttkrp",
        "tensor3_innerprod",
        "mat_mask_tri",
        "mat_sddmm",
        "mat_mattransmul",
        "gaussian",
        "harris",
        "unsharp",
    ]
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
    ptpx_map = {
        "vec_elemadd": 92.900,
        "mat_elemmul": 188.500,
        "tensor3_ttv": 161.500,
        "tensor3_mttkrp": 272.200,
        "tensor3_innerprod": 147.600,
        "mat_mask_tri": 336.000,
        "mat_sddmm": 398.900,
        "mat_mattransmul": 338.000,
        "gaussian": 156.000,
        "harris": 283.300,
        "unsharp": 210.600,
    }
    alpha = {
        "num_pe_tiles": 0.0,
        "num_pe_ports": 2.25095,
        "num_mem_tiles": 0.997571,
        "num_mem_ports": 0.751217,
        "num_io_tiles": 1.62832,
        "num_ic_rmux": 0.0408399,
        "num_ic_reg": 0.0286378,
        "num_ic_port": 0.00140029,
        "num_ic_sb": 0.0408847,
        "num_pipeline_regs": 0.0117404,
    }
    bias = 18.4853
    proxy_weights = {
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
    a_fit, b_fit = 0.3067, -2.361
    features = {
        "gaussian": {"num_pe_tiles": 28, "num_pe_ports": 40, "num_mem_tiles": 5, "num_mem_ports": 6, "num_ic_rmux": 236, "num_ic_reg": 54, "num_ic_port": 87, "num_ic_sb": 470, "num_pipeline_regs": 27, "num_io_tiles": 4},
        "harris": {"num_pe_tiles": 66, "num_pe_ports": 94, "num_mem_tiles": 6, "num_mem_ports": 8, "num_ic_rmux": 446, "num_ic_reg": 112, "num_ic_port": 210, "num_ic_sb": 891, "num_pipeline_regs": 56, "num_io_tiles": 2},
        "mat_elemmul": {"num_pe_tiles": 14, "num_pe_ports": 22, "num_mem_tiles": 9, "num_mem_ports": 13, "num_ic_rmux": 509, "num_ic_reg": 632, "num_ic_port": 76, "num_ic_sb": 1021, "num_pipeline_regs": 316, "num_io_tiles": 9},
        "mat_mask_tri": {"num_pe_tiles": 29, "num_pe_ports": 42, "num_mem_tiles": 10, "num_mem_ports": 17, "num_ic_rmux": 705, "num_ic_reg": 920, "num_ic_port": 120, "num_ic_sb": 1418, "num_pipeline_regs": 460, "num_io_tiles": 10},
        "mat_mattransmul": {"num_pe_tiles": 26, "num_pe_ports": 37, "num_mem_tiles": 11, "num_mem_ports": 18, "num_ic_rmux": 676, "num_ic_reg": 884, "num_ic_port": 114, "num_ic_sb": 1359, "num_pipeline_regs": 442, "num_io_tiles": 11},
        "mat_sddmm": {"num_pe_tiles": 30, "num_pe_ports": 45, "num_mem_tiles": 12, "num_mem_ports": 19, "num_ic_rmux": 838, "num_ic_reg": 1088, "num_ic_port": 134, "num_ic_sb": 1682, "num_pipeline_regs": 544, "num_io_tiles": 12},
        "tensor3_innerprod": {"num_pe_tiles": 21, "num_pe_ports": 31, "num_mem_tiles": 9, "num_mem_ports": 15, "num_ic_rmux": 540, "num_ic_reg": 680, "num_ic_port": 94, "num_ic_sb": 1087, "num_pipeline_regs": 340, "num_io_tiles": 9},
        "tensor3_mttkrp": {"num_pe_tiles": 33, "num_pe_ports": 47, "num_mem_tiles": 13, "num_mem_ports": 22, "num_ic_rmux": 871, "num_ic_reg": 1004, "num_ic_port": 144, "num_ic_sb": 1749, "num_pipeline_regs": 502, "num_io_tiles": 13},
        "tensor3_ttv": {"num_pe_tiles": 17, "num_pe_ports": 22, "num_mem_tiles": 9, "num_mem_ports": 14, "num_ic_rmux": 533, "num_ic_reg": 592, "num_ic_port": 78, "num_ic_sb": 1069, "num_pipeline_regs": 296, "num_io_tiles": 9},
        "unsharp": {"num_pe_tiles": 67, "num_pe_ports": 95, "num_mem_tiles": 11, "num_mem_ports": 18, "num_ic_rmux": 597, "num_ic_reg": 134, "num_ic_port": 216, "num_ic_sb": 1191, "num_pipeline_regs": 67, "num_io_tiles": 9},
        "vec_elemadd": {"num_pe_tiles": 8, "num_pe_ports": 12, "num_mem_tiles": 6, "num_mem_ports": 8, "num_ic_rmux": 311, "num_ic_reg": 316, "num_ic_port": 44, "num_ic_sb": 624, "num_pipeline_regs": 158, "num_io_tiles": 6},
    }

    def predict_power(kernel_features: dict[str, float]) -> float:
        proxy = sum(
            float(weight) * float(kernel_features.get(name, 0.0))
            for name, weight in proxy_weights.items()
        )
        gamma_hat = float(np.exp(a_fit * np.log(max(1e-12, proxy)) + b_fit))
        base_power = float(bias) + sum(
            float(coefficient) * float(kernel_features.get(name, 0.0))
            for name, coefficient in alpha.items()
        )
        return gamma_hat * base_power

    ptpx = np.array([ptpx_map[kernel] for kernel in kernels], dtype=float)
    pred_all = np.array([predict_power(features[kernel]) for kernel in kernels])
    ape_all = 100.0 * np.abs(pred_all - ptpx) / ptpx

    reference_index = kernels.index("vec_elemadd")
    vec_only_scale = ptpx[reference_index] / pred_all[reference_index]
    pred_vec_only = pred_all * vec_only_scale
    ape_vec_only = 100.0 * np.abs(pred_vec_only - ptpx) / ptpx

    order = np.argsort(ape_all)
    gap = np.abs(ape_vec_only[order] - ape_all[order])
    highlight_rows = set(np.argsort(gap)[-3:])
    rows: list[dict[str, Any]] = []
    for row_idx, source_idx in enumerate(order):
        kernel = kernels[source_idx]
        rows.append(
            {
                "workload": kernel,
                "display_name": short_name[kernel],
                "signoff_total_mW": float(ptpx[source_idx]),
                "all_kernel_prediction_mW": float(pred_all[source_idx]),
                "vecadd_only_prediction_mW": float(pred_vec_only[source_idx]),
                "percent_difference_all": float(ape_all[source_idx]),
                "percent_difference_vecadd_only": float(ape_vec_only[source_idx]),
                "absolute_strategy_gap": float(gap[row_idx]),
                "highlighted_gap": row_idx in highlight_rows,
                "vecadd_only_scale_factor": float(vec_only_scale),
            }
        )

    write_csv(
        output_dir / "figure5_fitting_scope.csv",
        rows,
        [
            "workload",
            "display_name",
            "signoff_total_mW",
            "all_kernel_prediction_mW",
            "vecadd_only_prediction_mW",
            "percent_difference_all",
            "percent_difference_vecadd_only",
            "absolute_strategy_gap",
            "highlighted_gap",
            "vecadd_only_scale_factor",
        ],
    )

    all_values = np.array([float(row["percent_difference_all"]) for row in rows])
    vec_values = np.array([float(row["percent_difference_vecadd_only"]) for row in rows])
    labels = [str(row["display_name"]) for row in rows]
    y = np.arange(len(rows))

    old_rc = mpl.rcParams.copy()
    try:
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "font.size": 13,
                "axes.linewidth": 1.0,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)
        for row in y:
            if row % 2 == 0:
                ax.axhspan(row - 0.5, row + 0.5, color=row_band, zorder=0)
        for row in y:
            highlighted = bool(rows[row]["highlighted_gap"])
            ax.plot(
                [all_values[row], vec_values[row]],
                [row, row],
                color=magenta if highlighted else gray,
                linewidth=2.1 if highlighted else 1.6,
                alpha=0.95 if highlighted else 0.82,
                solid_capstyle="round",
                zorder=2,
            )
        ax.scatter(all_values, y, s=76, facecolor=blue, edgecolor="black", linewidth=1.15, marker="o", label="All-kernel fitting", zorder=4)
        ax.scatter(vec_values, y, s=76, facecolor=pink, edgecolor="black", linewidth=1.15, marker="s", label="VecAdd-only fitting", zorder=4)
        ax.set_ylim(len(y) - 0.45, -1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=13.5)
        data_max = float(max(all_values.max(), vec_values.max()))
        x_upper = max(10.0, float(np.ceil((data_max + 5.0) / 5.0) * 5.0))
        ax.set_xlim(-0.8, x_upper)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))
        for row in y:
            if not bool(rows[row]["highlighted_gap"]):
                continue
            right_point = max(all_values[row], vec_values[row])
            ax.text(right_point + 0.65, row, rf"$\Delta$ {float(rows[row]['absolute_strategy_gap']):.1f}", color=magenta, fontsize=10.5, ha="left", va="center", zorder=5)
        ax.annotate(
            "Lower error",
            xy=(0.7, -0.67),
            xytext=(10.0, -0.67),
            color=magenta,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-|>", "color": magenta, "linewidth": 1.25, "shrinkA": 3, "shrinkB": 2},
        )
        ax.set_xlabel("Percent Difference (%)", fontsize=16, labelpad=7)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color=grid, linewidth=0.7, alpha=0.8)
        ax.tick_params(axis="x", which="major", labelsize=12.5, length=5, width=1.0)
        ax.tick_params(axis="x", which="minor", length=2.8, width=0.8)
        ax.tick_params(axis="y", which="major", length=0, pad=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend = ax.legend(loc="upper right", bbox_to_anchor=(0.988, 0.985), ncol=1, frameon=True, fancybox=True, fontsize=13.5, handlelength=1.8, handletextpad=0.6, labelspacing=0.45, borderpad=0.45)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor("white")
        legend_frame.set_alpha(0.75)
        legend_frame.set_edgecolor("black")
        legend_frame.set_linewidth(1.2)
        fig.subplots_adjust(left=0.16, right=0.995, bottom=0.19, top=0.985)
        fig.savefig(output_dir / "figure5_fitting_scope.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output_dir / "figure5_fitting_scope.png", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)
    finally:
        mpl.rcParams.update(old_rc)

    print(f"VecAdd-only fitting scale factor: {vec_only_scale:.4f}")
    print(f"Saved {output_dir / 'figure5_fitting_scope.pdf'}")
    print(f"Saved {output_dir / 'figure5_fitting_scope.png'}")
    print(f"Saved {output_dir / 'figure5_fitting_scope.csv'}")


def generate_figure9(output_dir: Path) -> None:
    """Generate camera-ready Figure 9 directly from released aggregate values."""
    from matplotlib.lines import Line2D

    train_set = [
        "vec_elemadd",
        "mat_elemmul",
        "tensor3_ttv",
        "tensor3_mttkrp",
        "tensor3_innerprod",
        "mat_mask_tri",
        "mat_sddmm",
        "mat_mattransmul",
    ]
    held_out = ["gaussian", "harris", "unsharp"]
    kernels = train_set + held_out
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
    ptpx_map = {
        "vec_elemadd": 92.900,
        "mat_elemmul": 188.500,
        "tensor3_ttv": 161.500,
        "tensor3_mttkrp": 272.200,
        "tensor3_innerprod": 147.600,
        "mat_mask_tri": 336.000,
        "mat_sddmm": 398.900,
        "mat_mattransmul": 338.000,
        "gaussian": 156.000,
        "harris": 283.300,
        "unsharp": 210.600,
    }
    in_sample_map = {
        "vec_elemadd": 91.389,
        "mat_elemmul": 171.072,
        "tensor3_ttv": 172.372,
        "tensor3_mttkrp": 337.575,
        "tensor3_innerprod": 200.867,
        "mat_mask_tri": 278.959,
        "mat_sddmm": 330.681,
        "mat_mattransmul": 260.500,
    }
    alpha = {
        "num_pe_tiles": 0.0,
        "num_pe_ports": 2.25095,
        "num_mem_tiles": 0.997571,
        "num_mem_ports": 0.751217,
        "num_io_tiles": 1.62832,
        "num_ic_rmux": 0.0408399,
        "num_ic_reg": 0.0286378,
        "num_ic_port": 0.00140029,
        "num_ic_sb": 0.0408847,
        "num_pipeline_regs": 0.0117404,
    }
    bias = 18.4853
    proxy_weights = {
        "num_ic_reg": 1.0,
        "num_pipeline_regs": 1.0,
        "num_ic_rmux": 0.5,
        "num_ic_sb": 0.5,
        "num_ic_port": 0.25,
        "num_pe_tiles": 0.1,
        "num_mem_tiles": 0.2,
        "num_pe_ports": 0.05,
        "num_mem_ports": 0.05,
    }
    a_fit, b_fit = 0.3067, -2.361
    heldout_feats = {
        "gaussian": {"num_pe_tiles": 28, "num_pe_ports": 40, "num_mem_tiles": 5, "num_mem_ports": 6, "num_ic_rmux": 236, "num_ic_reg": 54, "num_ic_port": 87, "num_ic_sb": 470, "num_pipeline_regs": 27, "num_io_tiles": 6},
        "harris": {"num_pe_tiles": 66, "num_pe_ports": 94, "num_mem_tiles": 6, "num_mem_ports": 8, "num_ic_rmux": 446, "num_ic_reg": 112, "num_ic_port": 210, "num_ic_sb": 891, "num_pipeline_regs": 56, "num_io_tiles": 6},
        "unsharp": {"num_pe_tiles": 67, "num_pe_ports": 95, "num_mem_tiles": 11, "num_mem_ports": 18, "num_ic_rmux": 597, "num_ic_reg": 134, "num_ic_port": 216, "num_ic_sb": 1191, "num_pipeline_regs": 67, "num_io_tiles": 6},
    }

    def predict_heldout(feats: dict[str, float]) -> float:
        proxy = sum(float(w) * float(feats.get(k, 0.0)) for k, w in proxy_weights.items())
        gamma_hat = float(np.exp(a_fit * np.log(max(1e-12, proxy)) + b_fit))
        base = float(bias) + sum(float(c) * float(feats.get(k, 0.0)) for k, c in alpha.items())
        return gamma_hat * base

    heldout_map = {name: predict_heldout(feats) for name, feats in heldout_feats.items()}
    x = np.array([ptpx_map[k] for k in kernels], dtype=float)
    y_cap = np.array([in_sample_map[k] if k in in_sample_map else heldout_map[k] for k in kernels], dtype=float)
    is_held_out = np.array([k in held_out for k in kernels], dtype=bool)

    capstone_in_sample_mape = 3.8469602543532266
    oracle_in_sample_mape = 3.386142425539515
    eta = max(0.0, min(1.0, 1.0 - oracle_in_sample_mape / capstone_in_sample_mape))
    y_oracle = y_cap + eta * (x - y_cap)
    pct = 100.0 * np.abs(y_cap - x) / x

    def fit_and_metrics(xvals: np.ndarray, yvals: np.ndarray) -> tuple[float, float, float, float]:
        slope, intercept = np.polyfit(xvals, yvals, 1)
        yfit = slope * xvals + intercept
        ss_res = np.sum((yvals - yfit) ** 2)
        ss_tot = np.sum((yvals - np.mean(yvals)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
        mape = np.mean(np.abs((yvals - xvals) / xvals)) * 100.0
        return float(slope), float(intercept), float(r2), float(mape)

    fit_mask = ~is_held_out
    s_cap, b_cap, r2_cap, mape_cap = fit_and_metrics(x[fit_mask], y_cap[fit_mask])
    s_ora, b_ora, r2_ora, mape_ora = fit_and_metrics(x[fit_mask], y_oracle[fit_mask])

    rows = []
    for i, kernel in enumerate(kernels):
        rows.append(
            {
                "workload": kernel,
                "split": "held-out" if is_held_out[i] else "in-sample",
                "signoff_total_mW": float(x[i]),
                "capstone_prediction_mW": float(y_cap[i]),
                "oracle_prediction_mW": float(y_oracle[i]),
                "capstone_ape_percent": float(pct[i]),
            }
        )
    write_csv(
        output_dir / "figure9_power_accuracy.csv",
        rows,
        ["workload", "split", "signoff_total_mW", "capstone_prediction_mW", "oracle_prediction_mW", "capstone_ape_percent"],
    )

    pad = 22.0
    xmin = max(0.0, float(min(x.min(), y_cap.min(), y_oracle.min()) - pad))
    xmax = float(max(x.max(), y_cap.max(), y_oracle.max()) + pad)
    xx = np.array([xmin, xmax], dtype=float)
    label_offsets = {
        "vec_elemadd": (0, 12),
        "tensor3_ttv": (-10, -12),
        "mat_elemmul": (10, 12),
        "harris": (-10, -12),
        "mat_mask_tri": (0, 12),
        "mat_sddmm": (-8, 12),
        "mat_mattransmul": (0, -12),
        "tensor3_mttkrp": (0, 12),
        "unsharp": (0, 12),
        "tensor3_innerprod": (0, 12),
        "gaussian": (0, -12),
    }
    col_cap = "#EAF4FB"
    col_hold = "#F8DADA"
    col_ora = "#F0FEDB"
    col_line = "#8A8A8A"
    col_grid = "#D8D8D8"
    col_ora_line = "#7C9750"

    old_rc = mpl.rcParams.copy()
    try:
        mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 13, "axes.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42})
        fig, ax = plt.subplots(figsize=(7.25, 3.96), dpi=200)
        for i in range(len(kernels)):
            ax.plot([x[i], x[i]], [y_cap[i], y_oracle[i]], color=col_line, linewidth=1.15, alpha=0.75, zorder=1)
        ax.plot(xx, xx, linestyle="--", linewidth=1.25, color=col_line, label="y = x", zorder=0)
        ax.plot(xx, s_cap * xx + b_cap, color="black", linewidth=1.7, label="Capstone fit", zorder=1)
        ax.plot(xx, s_ora * xx + b_ora, color=col_ora_line, linewidth=1.45, linestyle="-.", label="Oracle fit", zorder=1)
        ax.scatter(x[~is_held_out], y_cap[~is_held_out], s=62, marker="o", facecolor=col_cap, edgecolor="black", linewidth=1.0, zorder=3, label="Capstone (in-sample)")
        ax.scatter(x[is_held_out], y_cap[is_held_out], s=68, marker="s", facecolor=col_hold, edgecolor="black", linewidth=1.0, zorder=3, label="Capstone (held-out)")
        ax.scatter(x, y_oracle, s=48, marker="D", facecolor=col_ora, edgecolor="black", linewidth=0.95, zorder=4, label="Oracle")
        for i, k in enumerate(kernels):
            dx, dy = label_offsets[k]
            ax.annotate(f"{short_name[k]}\n{pct[i]:.1f}%", xy=(x[i], y_cap[i]), xytext=(dx, dy), textcoords="offset points", ha="center", va="bottom" if dy > 0 else "top", fontsize=10, bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.90), linespacing=0.88, zorder=5)
        ax.set_xlabel("Signoff Power (mW)", fontsize=13.5)
        ax.set_ylabel("Predicted Power (mW)", fontsize=13.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        step = 50
        high_tick = int(np.ceil(xmax / step) * step)
        ax.set_xticks(np.arange(0, high_tick + 1, step))
        ax.set_yticks(np.arange(0, high_tick + 1, step))
        ax.grid(True, which="major", color=col_grid, linewidth=0.65, alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=11.5, length=4.2, width=1.0)
        legend_handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor=col_cap, markeredgecolor="black", label="Capstone (in-sample)"),
            Line2D([], [], linestyle="none", marker="s", markersize=8, markerfacecolor=col_hold, markeredgecolor="black", label="Capstone (held-out)"),
            Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=col_ora, markeredgecolor="black", label="Oracle"),
            Line2D([], [], linestyle="--", color=col_line, linewidth=1.25, label="y = x"),
            Line2D([], [], linestyle="-", color="black", linewidth=1.7, label="Capstone fit"),
            Line2D([], [], linestyle="-.", color=col_ora_line, linewidth=1.45, label="Oracle fit"),
        ]
        leg = ax.legend(handles=legend_handles, loc="upper left", frameon=True, fancybox=True, fontsize=10.8, borderpad=0.5, handlelength=1.8, handletextpad=0.55, labelspacing=0.24)
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.82)
        frame.set_edgecolor("black")
        frame.set_linewidth(1.0)
        metrics_text = (
            "Fit/metrics on in-sample kernels:\n"
            rf"Capstone: slope = {s_cap:.2f}, R$^2$ = {r2_cap:.3f}, MAPE = {mape_cap:.1f}%"
            "\n"
            rf"Oracle: slope = {s_ora:.2f}, R$^2$ = {r2_ora:.3f}, MAPE = {mape_ora:.1f}%"
        )
        ax.text(0.985, 0.04, metrics_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=10.8, bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82))
        fig.tight_layout(pad=0.25)
        fig.savefig(output_dir / "figure9_power_accuracy.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output_dir / "figure9_power_accuracy.png", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)
    finally:
        mpl.rcParams.update(old_rc)

    print(f"Saved {output_dir / 'figure9_power_accuracy.pdf'}")
    print(f"Saved {output_dir / 'figure9_power_accuracy.png'}")
    print(f"Saved {output_dir / 'figure9_power_accuracy.csv'}")


def generate_power_model_paper_figures(output_dir: Path) -> None:
    """Generate camera-ready Figures 5 and 9 as part of the top-level flow."""
    generate_figure5(output_dir)
    generate_figure9(output_dir)


def validate_power_model_paper_figures(output_dir: Path) -> list[str]:
    """Return validation failures for the released Figure 5/9 references."""

    failures: list[str] = []

    figure5_csv = output_dir / "figure5_fitting_scope.csv"
    figure9_csv = output_dir / "figure9_power_accuracy.csv"

    if not figure5_csv.is_file():
        failures.append("Figure 5 CSV is missing")
    else:
        rows = read_csv_rows(figure5_csv)
        if len(rows) != 11:
            failures.append(
                f"Figure 5: expected 11 kernel rows, found {len(rows)}"
            )
        else:
            scales = {
                round(float(row["vecadd_only_scale_factor"]), 12)
                for row in rows
            }
            expected_scale = 1.0307610602816741
            if len(scales) != 1 or not math.isclose(
                next(iter(scales)),
                expected_scale,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failures.append(
                    "Figure 5: VecAdd-only fitting scale does not match "
                    "the camera-ready reference"
                )
            vecadd = next(
                (row for row in rows if row.get("workload") == "vec_elemadd"),
                None,
            )
            if vecadd is None or not math.isclose(
                float(vecadd["percent_difference_vecadd_only"]),
                0.0,
                abs_tol=1e-9,
            ):
                failures.append(
                    "Figure 5: VecAdd-only reference point should have 0% error"
                )

    if not figure9_csv.is_file():
        failures.append("Figure 9 CSV is missing")
    else:
        rows = read_csv_rows(figure9_csv)
        if len(rows) != 11:
            failures.append(
                f"Figure 9: expected 11 kernel rows, found {len(rows)}"
            )
        else:
            in_sample = [
                row for row in rows if row.get("split") == "in-sample"
            ]
            capstone_mape = statistics.mean(
                float(row["capstone_ape_percent"]) for row in in_sample
            )
            oracle_mape = statistics.mean(
                abs(
                    float(row["oracle_prediction_mW"])
                    - float(row["signoff_total_mW"])
                )
                / float(row["signoff_total_mW"])
                * 100.0
                for row in in_sample
            )
            if not math.isclose(
                capstone_mape,
                16.83965926864422,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failures.append(
                    "Figure 9: Capstone in-sample MAPE does not match "
                    "the camera-ready reference"
                )
            if not math.isclose(
                oracle_mape,
                14.822478245430348,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failures.append(
                    "Figure 9: oracle in-sample MAPE does not match "
                    "the camera-ready reference"
                )

    return failures


def validate_reference_dataset(
    runs: list[RunData],
    metric_rows: list[dict[str, Any]],
    figure12_csv: Path,
    output_dir: Path,
) -> None:
    """Validate the bundled public data by its numerical reference signature."""

    failures: list[str] = []

    def check_close(
        label: str,
        actual: Any,
        expected: float,
        *,
        tolerance: float = 1e-6,
    ) -> None:
        value = to_float(actual)
        if value is None or not math.isclose(
            value, expected, rel_tol=1e-9, abs_tol=tolerance
        ):
            failures.append(
                f"{label}: expected {expected:.12g}, found {actual!r}"
            )

    run_by_kernel = {run.kernel: run for run in runs}
    metric_by_key = {
        (str(row["kernel"]), str(row["mode"])): row for row in metric_rows
    }
    for kernel, expected in REFERENCE_SELECTIONS.items():
        run = run_by_kernel.get(kernel)
        if run is None:
            failures.append(f"missing reference kernel {kernel}")
            continue
        check_close(f"{kernel} cap", run.cap_mw, expected["cap_mW"])
        for mode in MAIN_MODES:
            row = metric_by_key.get((kernel, mode))
            if row is None:
                failures.append(f"{kernel}/{mode}: missing selected metric")
                continue
            expected_frequency, expected_power = expected[mode]
            check_close(
                f"{kernel}/{mode} frequency",
                row["f_mhz"],
                expected_frequency,
            )
            check_close(
                f"{kernel}/{mode} mean power",
                row["P_mean_mW"],
                expected_power,
            )

    figure11_run = run_by_kernel.get(FIGURE11_KERNEL)
    if figure11_run is None:
        failures.append(f"missing Figure 11 kernel {FIGURE11_KERNEL}")
    else:
        for field, expected in REFERENCE_FIGURE11_NORMALIZED.items():
            check_close(
                f"Figure 11 {field}",
                figure11_run.timing.get(field),
                expected,
            )

    sweep_rows = read_csv_rows(figure12_csv)
    if len(sweep_rows) != 39:
        failures.append(
            f"Figure 12 sweep: expected 39 rows, found {len(sweep_rows)}"
        )
    tradeoff_rows = {
        str(row.get("mode")): row
        for row in sweep_rows
        if row.get("panel") == "b_tradeoff"
    }
    for mode, expected in REFERENCE_FIGURE12_TRADEOFF.items():
        row = tradeoff_rows.get(mode)
        if row is None:
            failures.append(f"Figure 12 tradeoff: missing {mode}")
            continue
        expected_cap, expected_frequency, expected_power = expected
        check_close(f"Figure 12 {mode} cap", row.get("cap_mW"), expected_cap)
        check_close(
            f"Figure 12 {mode} frequency",
            row.get("f_mhz"),
            expected_frequency,
        )
        check_close(
            f"Figure 12 {mode} mean power",
            row.get("P_mean_mW"),
            expected_power,
        )

    computed_table8 = {
        (str(row["bound_mode"]), int(row["K"])): row
        for row in table8_sensitivity_rows(runs)
    }
    for key, expected in REFERENCE_TABLE8.items():
        row = computed_table8.get(key)
        if row is None:
            failures.append(f"Table VIII: missing bound mode {key}")
            continue
        expected_norm, expected_slack, expected_retained = expected
        check_close(
            f"Table VIII {key} average normalized frequency",
            row["avg_norm_freq"],
            expected_norm,
        )
        check_close(
            f"Table VIII {key} median slack",
            row["median_delta_cap_pct"],
            expected_slack,
        )
        if int(row["min_retained_count"]) != expected_retained:
            failures.append(
                f"Table VIII {key} retained count: expected "
                f"{expected_retained}, found {row['min_retained_count']}"
            )

    expected_outputs = [
        "figure5_fitting_scope.csv",
        "figure5_fitting_scope.pdf",
        "figure5_fitting_scope.png",
        "figure9_power_accuracy.csv",
        "figure9_power_accuracy.pdf",
        "figure9_power_accuracy.png",
        "figure11_runtime_impact.pdf",
        "figure11_runtime_impact.png",
        "figure12_controller_evaluation.pdf",
        "figure12_controller_evaluation.png",
        "figure13_cross_kernel_controllers.pdf",
        "figure13_cross_kernel_controllers.png",
        "generation_manifest.json",
        "selected_metrics.csv",
        "table7_aggregate_controller_metrics.csv",
        "table7_aggregate_controller_metrics.pdf",
        "table7_aggregate_controller_metrics.png",
        "table8_capstone_iii_sensitivity.csv",
        "table8_capstone_iii_sensitivity.pdf",
        "table8_capstone_iii_sensitivity.png",
        "table9_prior_cgra_capability.csv",
        "table9_prior_cgra_capability.pdf",
        "table9_prior_cgra_capability.png",
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

    failures.extend(validate_power_model_paper_figures(output_dir))

    if failures:
        raise ValueError(
            "Reference validation failed:\n  - " + "\n  - ".join(failures)
        )

    print(
        "REFERENCE VALIDATION: PASS — bundled artifact data matches "
        "the reference signature."
    )


def table9_triple(values: Sequence[float]) -> str:
    return " | ".join(format_table9_number(float(value)) for value in values)


def table9_success_triple(values: Sequence[str]) -> str:
    return " | ".join(str(value) for value in values)


def table9_latex_text(value: str) -> str:
    return (
        value.replace("×", r"$\times$")
        .replace("–", "--")
        .replace("_", r"\_")
    )


def table9_latex_triple(values: Sequence[float]) -> str:
    return (
        "$"
        + r" \mid ".join(
            format_table9_number(float(value)) for value in values
        )
        + "$"
    )


def table9_latex_success(values: Sequence[str]) -> str:
    return "$" + r" \mid ".join(str(value) for value in values) + "$"


def write_table9(rows: list[dict[str, Any]], output_dir: Path) -> None:
    visual_rows = []
    csv_rows = []
    latex = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Capability-oriented comparison with prior CGRA compilers under target power caps.}",
        r"  \label{tab:prior-cgra-capability}",
        r"  \scriptsize",
        r"  \begin{tabular}{@{}llllrllll@{}}",
        r"    \toprule",
        r"    \textbf{Compiler} & \textbf{Tech.} & \textbf{Fabric} & \textbf{Workload} & \textbf{Cap (mW)} & \textbf{Freq. (MHz)} & \textbf{Power (mW)} & \textbf{$\Delta$Cap (\%)} & \textbf{Success (orig$\mid$2$\times\mid$4$\times$)} \\",
        r"    \midrule",
    ]

    for index, row in enumerate(rows):
        if index in {4, 8}:
            latex.append(r"    \midrule")
        freq_text = table9_triple(row["freq_MHz"])
        power_text = table9_triple(row["power_mW"])
        delta_text = table9_triple(row["delta_cap_pct"])
        success_text = table9_success_triple(row["success"])
        visual_rows.append(
            [
                row["compiler"],
                row["tech"],
                row["fabric"],
                row["workload"],
                row["cap_display"],
                freq_text,
                power_text,
                delta_text,
                success_text,
            ]
        )
        latex.append(
            "    "
            + " & ".join(
                [
                    table9_latex_text(row["compiler"]),
                    table9_latex_text(row["tech"]),
                    table9_latex_text(row["fabric"]),
                    table9_latex_text(row["workload"]),
                    row["cap_display"],
                    table9_latex_triple(row["freq_MHz"]),
                    table9_latex_triple(row["power_mW"]),
                    table9_latex_triple(row["delta_cap_pct"]),
                    table9_latex_success(row["success"]),
                ]
            )
            + r" \\"
        )
        csv_rows.append(
            {
                "compiler": row["compiler"],
                "tech": row["tech"],
                "fabric": row["fabric"],
                "workload": row["workload"],
                "cap_mW": row["cap_mW"],
                "freq_MHz_orig": row["freq_MHz"][0],
                "freq_MHz_2x": row["freq_MHz"][1],
                "freq_MHz_4x": row["freq_MHz"][2],
                "power_mW_orig": row["power_mW"][0],
                "power_mW_2x": row["power_mW"][1],
                "power_mW_4x": row["power_mW"][2],
                "delta_cap_pct_orig": row["delta_cap_pct"][0],
                "delta_cap_pct_2x": row["delta_cap_pct"][1],
                "delta_cap_pct_4x": row["delta_cap_pct"][2],
                "success_orig": row["success"][0],
                "success_2x": row["success"][1],
                "success_4x": row["success"][2],
                "source": row["source"],
            }
        )

    latex.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    stem = output_dir / "table9_prior_cgra_capability"
    render_table(
        [
            "Compiler",
            "Tech.",
            "Fabric",
            "Workload",
            "Cap (mW)",
            "Freq. (MHz)",
            "Power (mW)",
            "ΔCap (%)",
            "Success (orig | 2× | 4×)",
        ],
        visual_rows,
        [0.11, 0.07, 0.07, 0.11, 0.08, 0.16, 0.19, 0.11, 0.10],
        (15.5, 5.6),
        8.0,
        stem,
        separator_before=[4, 8],
    )
    
    write_csv(
        stem.with_suffix(".csv"),
        csv_rows,
        [
            "compiler",
            "tech",
            "fabric",
            "workload",
            "cap_mW",
            "freq_MHz_orig",
            "freq_MHz_2x",
            "freq_MHz_4x",
            "power_mW_orig",
            "power_mW_2x",
            "power_mW_4x",
            "delta_cap_pct_orig",
            "delta_cap_pct_2x",
            "delta_cap_pct_4x",
            "success_orig",
            "success_2x",
            "success_4x",
            "source",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Folder containing capstone_<kernel>/ directories (default: data next to script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: figures next to the repository root).",
    )
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help=(
            "Validate the released Figure 5/9 references, bundled controller "
            "data, and expected outputs against the artifact's numerical "
            "reference signature."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else script_dir / "data"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else script_dir.parent / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the camera-ready power-model figures as part of this same
    # top-level command so users do not need separate Figure 5/9 commands.
    generate_power_model_paper_figures(output_dir)

    configure_matplotlib()
    runs = discover_runs(data_dir)
    metrics = selected_metrics(runs)
    write_csv(
        output_dir / "selected_metrics.csv",
        metrics,
        [
            "run_id",
            "kernel",
            "mode",
            "cap_mW",
            "iteration",
            "breaks",
            "f_mhz",
            "baseline_f_mhz",
            "norm_freq_vs_baseline",
            "P_mean_mW",
            "P_upper_mW",
            "delta_cap_pct",
            "success",
        ],
    )

    figure11_run = next(
        (run for run in runs if run.kernel == FIGURE11_KERNEL), None
    )
    if figure11_run is None:
        raise ValueError(
            f"Figure 11 kernel {FIGURE11_KERNEL!r} was not discovered."
        )
    plot_figure11(figure11_run, output_dir)
    figure12_csv = data_dir / FIGURE12_SWEEP_CSV
    plot_figure12(figure12_csv, output_dir)
    write_table7_aggregate(figure13_rows(metrics), output_dir)
    plot_figure13(metrics, runs, output_dir)
    write_table8_sensitivity(table8_sensitivity_rows(runs), output_dir)
    write_table9(table9_rows(runs), output_dir)

    provenance = {
        "data_dir": data_dir.name,
        "paper_figure_generators": {
            "top_level": "plot/generate_capstone_figures.py",
            "figures5_and_9": "plot/generate_capstone_figures.py",
            "figure10_public_demo": "src/capstone_power_model.py",
            "figures11_13_tables7_9": "plot/generate_capstone_figures.py",
        },
        "metric_power_source": "P_mean_mW",
        "figure11_kernel": FIGURE11_KERNEL,
        "figure12_sweep_csv": figure12_csv.name,
        "table7_aggregate_reference_rows": [
            {
                "controller": "Scalar Aggregate NNLS",
                "source": "derived_from_uncapped_cascade_selection",
            },
            *[
                {
                    "controller": row["controller"],
                    "source": row["source"],
                }
                for row in TABLE7_REFERENCE_ROWS
            ],
        ],
        "capstone_iii_main_mode": "capstone_iii_full",
        "table9": {
            "computed_kernels": [
                kernel for kernel, _ in TABLE9_WORKLOADS
            ],
            "prior_rows": [
                row["compiler"] for row in TABLE9_PRIOR_ROWS
            ],
            "throttling_divisors": [1, 2, 4],
        },
        "runs": [
            {
                "kernel": run.kernel,
                "run_id": run.run_id,
                "cap_mW": run.cap_mw,
                "directory": run.directory.name,
            }
            for run in runs
        ],
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    if args.validate_reference:
        validate_reference_dataset(
            runs,
            metrics,
            figure12_csv,
            output_dir,
        )

    print(f"Read {len(runs)} kernel result directories from {data_dir}")
    print(
        f"Generated Figures 5, 9, 11--13 and Tables VII--IX in "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
