#!/usr/bin/env python3
"""
Build the CSV file used to plot Capstone Figure 12.

First use pipeline.py inside the AHA Tutorial Docker container to capture one
complete tensor3_ttv run with a deliberately unreachable cap. This keeps every
controller active so its upper bound is logged for every pipeline candidate.
For the cap-normalized Capstone II engineering rule, use:

    export CAPSTONE_RUN_ID=tensor3_ttv_high_cap
    export CAPSTONE_POWER_CAP_MW=1000000000
    export CAPSTONE_II_ANCHOR_Q_MW=64000000
    export CAPSTONE_II_SPEC_Q_MW=40000000
    ./cascade_demo.sh max

Copy capstone_all_modes_trace.csv to:

    data/capstone_tensor3_ttv_high_cap/capstone_all_modes_trace.csv

Then run this script locally:

    python3 generate_figure12_sweep_data.py

It writes:

    data/figure12_sweep.csv

The CSV contains the Figure 12 cap-sweep points and a compact set of selected
700 mW points used for the slack-to-cap annotation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_CAPS_MW = [
    500.0,
    550.0,
    600.0,
    650.0,
    700.0,
    750.0,
    800.0,
    900.0,
    1000.0,
]

DEFAULT_TRADEOFF_CAPS_MW = {
    "capstone_i": 700.0,
    "capstone_ii": 700.0,
    "capstone_iii_full": 700.0,
}

MODES = [
    "baseline",
    "capstone_i",
    "capstone_ii",
    "capstone_iii_full",
]

MODE_LABELS = {
    "baseline": "Baseline (no capping)",
    "capstone_i": "Capstone I",
    "capstone_ii": "Capstone II",
    "capstone_iii_full": "Capstone III",
}

OUTPUT_FIELDS = [
    "panel",
    "run_id",
    "kernel",
    "mode",
    "mode_label",
    "cap_mW",
    "safe_candidate_found",
    "iteration",
    "breaks",
    "f_mhz",
    "P_mean_mW",
    "P_selection_upper_mW",
    "delta_cap_pct",
    "selection_rule",
    "capstone_ii_anchor_fraction",
    "freq_ref_mhz",
]


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


def parse_caps(raw_value: str) -> list[float]:
    values = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError(f"Cap must be positive, got {value}.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one cap is required.")
    return sorted(values)


def read_trace(
    trace_path: Path, requested_run_id: str | None
) -> tuple[str, dict[str, list[dict[str, str]]]]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace CSV does not exist: {trace_path}")

    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Trace CSV is empty: {trace_path}")

    run_ids = sorted(
        {
            str(row.get("run_id", "")).strip()
            for row in rows
            if str(row.get("run_id", "")).strip()
        }
    )
    if requested_run_id:
        run_id = requested_run_id
        if run_id not in run_ids:
            raise ValueError(
                f"run_id={run_id!r} is not present in {trace_path}. "
                f"Available run_ids: {run_ids}"
            )
    else:
        if len(run_ids) != 1:
            raise ValueError(
                "The trace contains multiple run_ids. Pass --run-id to select "
                "one of them."
            )
        run_id = run_ids[0]

    rows = [row for row in rows if row.get("run_id") == run_id]
    rows_by_mode = {
        mode: [row for row in rows if row.get("mode") == mode]
        for mode in MODES
    }
    missing = [mode for mode, mode_rows in rows_by_mode.items() if not mode_rows]
    if missing:
        raise ValueError(
            "The trace is missing required modes: " + ", ".join(missing)
        )

    # A normal capped run stops logging a controller after its first crossing.
    # Figure 12 requires every controller to reach the baseline's last candidate.
    last_baseline_iteration = max(
        int(to_int(row.get("iteration"), -1))
        for row in rows_by_mode["baseline"]
    )
    truncated = []
    for mode in MODES[1:]:
        last_mode_iteration = max(
            int(to_int(row.get("iteration"), -1))
            for row in rows_by_mode[mode]
        )
        if last_mode_iteration != last_baseline_iteration:
            truncated.append(
                f"{mode} ends at iteration {last_mode_iteration}, "
                f"baseline ends at {last_baseline_iteration}"
            )
    if truncated:
        raise ValueError(
            "This is not a complete high-cap trajectory. "
            + "; ".join(truncated)
            + ". Rerun tensor3_ttv with an unreachable cap so all modes remain "
            "active through maximal pipelining."
        )
    return run_id, rows_by_mode


def candidate_upper_mw(
    row: dict[str, str],
    mode: str,
    cap_mw: float,
    capstone_ii_anchor_fraction: float,
    freq_ref_mhz: float,
    scale_capstone_ii_by_frequency: bool,
) -> tuple[float, str]:
    if mode == "baseline":
        return float(to_float(row["P_mean_mW"])), "uncapped"

    if mode == "capstone_i":
        upper = to_float(row.get("P_upper_anchor_mW"))
        if upper is None:
            raise ValueError(
                "Capstone I trace rows need P_upper_anchor_mW."
            )
        return float(upper), "guardband anchor upper"

    if mode == "capstone_ii":
        f_mhz = float(to_float(row["f_mhz"]))
        rho = (
            max(1.0, f_mhz / freq_ref_mhz)
            if scale_capstone_ii_by_frequency
            else 1.0
        )
        mean_mw = float(to_float(row["P_mean_mW"]))
        upper = mean_mw + capstone_ii_anchor_fraction * cap_mw * rho
        return (
            upper,
            "cap-normalized anchor margin with frequency scaling"
            if scale_capstone_ii_by_frequency
            else "cap-normalized anchor margin",
        )

    if mode == "capstone_iii_full":
        upper = to_float(row.get("P_upper_robust_mW"))
        if upper is None:
            raise ValueError(
                "Full-bounds Capstone III trace rows need "
                "P_upper_robust_mW."
            )
        return float(upper), "full bounded-error robust upper"

    raise ValueError(f"Unsupported mode: {mode}")


def select_candidate(
    rows_by_mode: dict[str, list[dict[str, str]]],
    mode: str,
    cap_mw: float,
    capstone_ii_anchor_fraction: float,
    freq_ref_mhz: float,
    scale_capstone_ii_by_frequency: bool,
) -> tuple[dict[str, str], float, str] | None:
    safe: list[tuple[dict[str, str], float, str]] = []
    for row in rows_by_mode[mode]:
        upper_mw, rule = candidate_upper_mw(
            row,
            mode,
            cap_mw,
            capstone_ii_anchor_fraction,
            freq_ref_mhz,
            scale_capstone_ii_by_frequency,
        )
        if mode == "baseline" or upper_mw <= cap_mw + 1e-9:
            safe.append((row, upper_mw, rule))

    if not safe:
        return None

    # The objective is to maximize frequency, then select the
    # candidate whose upper envelope lands closest to the cap.
    return max(
        safe,
        key=lambda item: (
            float(to_float(item[0].get("f_mhz"), -math.inf)),
            item[1],
            int(to_int(item[0].get("iteration"), -1)),
        ),
    )


def result_row(
    panel: str,
    run_id: str,
    kernel: str,
    mode: str,
    cap_mw: float,
    selected: tuple[dict[str, str], float, str] | None,
    capstone_ii_anchor_fraction: float,
    freq_ref_mhz: float,
) -> dict[str, Any]:
    if selected is None:
        return {
            "panel": panel,
            "run_id": run_id,
            "kernel": kernel,
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "cap_mW": cap_mw,
            "safe_candidate_found": False,
            "iteration": "",
            "breaks": "",
            "f_mhz": "",
            "P_mean_mW": "",
            "P_selection_upper_mW": "",
            "delta_cap_pct": "",
            "selection_rule": "",
            "capstone_ii_anchor_fraction": capstone_ii_anchor_fraction,
            "freq_ref_mhz": freq_ref_mhz,
        }

    candidate, upper_mw, rule = selected
    mean_mw = float(to_float(candidate["P_mean_mW"]))
    return {
        "panel": panel,
        "run_id": run_id,
        "kernel": kernel,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "cap_mW": cap_mw,
        "safe_candidate_found": True,
        "iteration": int(to_int(candidate["iteration"])),
        "breaks": int(to_int(candidate["breaks"])),
        "f_mhz": float(to_float(candidate["f_mhz"])),
        "P_mean_mW": mean_mw,
        "P_selection_upper_mW": upper_mw,
        "delta_cap_pct": (
            "" if mode == "baseline" else 100.0 * (cap_mw - mean_mw) / cap_mw
        ),
        "selection_rule": rule,
        "capstone_ii_anchor_fraction": capstone_ii_anchor_fraction,
        "freq_ref_mhz": freq_ref_mhz,
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=(
            script_dir
            / "data"
            / "capstone_tensor3_ttv_high_cap"
            / "capstone_all_modes_trace.csv"
        ),
        help="Complete high-cap tensor3_ttv trace CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "data" / "figure12_sweep.csv",
        help="Output CSV containing the Figure 12 sweep and 700 mW annotation points.",
    )
    parser.add_argument(
        "--run-id",
        help="Trace run_id. It is inferred when the trace contains one run.",
    )
    parser.add_argument("--kernel", default="tensor3_ttv")
    parser.add_argument(
        "--caps",
        default=",".join(f"{value:g}" for value in DEFAULT_CAPS_MW),
        help="Comma-separated caps for the Figure 12 sweep.",
    )
    parser.add_argument(
        "--tradeoff-cap-i",
        type=float,
        default=DEFAULT_TRADEOFF_CAPS_MW["capstone_i"],
    )
    parser.add_argument(
        "--tradeoff-cap-ii",
        type=float,
        default=DEFAULT_TRADEOFF_CAPS_MW["capstone_ii"],
    )
    parser.add_argument(
        "--tradeoff-cap-iii",
        type=float,
        default=DEFAULT_TRADEOFF_CAPS_MW["capstone_iii_full"],
    )
    parser.add_argument(
        "--capstone-ii-anchor-fraction",
        type=float,
        default=0.064,
        help="q_anchor / cap for the Capstone II engineering demonstration.",
    )
    parser.add_argument("--freq-ref-mhz", type=float, default=100.0)
    parser.add_argument(
        "--no-capstone-ii-frequency-scaling",
        action="store_true",
        help="Use rho(f)=1 instead of max(1, f/f_ref).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = args.trace.resolve()
    output_path = args.output.resolve()
    caps_mw = parse_caps(args.caps)

    if args.capstone_ii_anchor_fraction < 0.0:
        raise ValueError("--capstone-ii-anchor-fraction cannot be negative.")
    if args.freq_ref_mhz <= 0.0:
        raise ValueError("--freq-ref-mhz must be positive.")
    tradeoff_caps = {
        "capstone_i": args.tradeoff_cap_i,
        "capstone_ii": args.tradeoff_cap_ii,
        "capstone_iii_full": args.tradeoff_cap_iii,
    }
    if any(value <= 0.0 for value in tradeoff_caps.values()):
        raise ValueError("Every tradeoff cap must be positive.")

    run_id, rows_by_mode = read_trace(trace_path, args.run_id)
    scale_by_frequency = not args.no_capstone_ii_frequency_scaling

    output_rows = []
    for cap_mw in caps_mw:
        for mode in MODES:
            selected = select_candidate(
                rows_by_mode,
                mode,
                cap_mw,
                args.capstone_ii_anchor_fraction,
                args.freq_ref_mhz,
                scale_by_frequency,
            )
            output_rows.append(
                result_row(
                    "a_cap_sweep",
                    run_id,
                    args.kernel,
                    mode,
                    cap_mw,
                    selected,
                    args.capstone_ii_anchor_fraction,
                    args.freq_ref_mhz,
                )
            )

    for mode in MODES[1:]:
        cap_mw = tradeoff_caps[mode]
        selected = select_candidate(
            rows_by_mode,
            mode,
            cap_mw,
            args.capstone_ii_anchor_fraction,
            args.freq_ref_mhz,
            scale_by_frequency,
        )
        if selected is None:
            raise ValueError(
                f"No safe {MODE_LABELS[mode]} candidate exists at "
                f"{cap_mw:g} mW for the Figure 12 slack annotation."
            )
        output_rows.append(
            result_row(
                "b_tradeoff",
                run_id,
                args.kernel,
                mode,
                cap_mw,
                selected,
                args.capstone_ii_anchor_fraction,
                args.freq_ref_mhz,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Read complete trajectory: {trace_path}")
    print(f"Run ID: {run_id}")
    print(f"Figure 12 sweep caps: {', '.join(f'{value:g}' for value in caps_mw)} mW")
    print(
        "Figure 12 annotation caps: "
        + ", ".join(
            f"{MODE_LABELS[mode]}={cap:g} mW"
            for mode, cap in tradeoff_caps.items()
        )
    )
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
