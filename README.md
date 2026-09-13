# Capstone MICRO 2026 Paper Artifact

- Title: Capstone: Power-Capped Pipelining for Coarse-Grained Reconfigurable Array Compilers
- Authors: Sabrina Yarzada and Christopher Torng
- Contact: noorahma@usc.edu

![**Capstone evaluation summary.** Left: predicted versus signoff PTPX power at 100 MHz for in-sample and held-out kernels, including the oracle reference and fitted trends. Right: normalized achievable frequency and remaining power-cap headroom across kernels for Capstone I--III, illustrating how the controller modes trade conservatism for performance under the power constraint. Please see the paper for more detailed information.](images/figure_README.png)

<p align="center">
  <em>
    <strong>Capstone evaluation highlights</strong>
    Left: Power estimation accuracy (Capstone prediction versus signoff).
    Right: Normalized frequency and power cap headroom for Capstone I-III across kernels.
    Please see the paper for additional details.
  </em>
</p>

This repository accompanies **“Capstone: Power-Capped Pipelining for Coarse-Grained Reconfigurable Array Compilers.”** It contains the Capstone power estimator, Capstone I-III controller logic, Cascade CGRA compiler integration, and scripts for generating Figures 5 and 9–13 and Tables VII–IX in the paper.

## Important scope and NDA notice

The Synopsys PrimeTime PX (PTPX) hierarchy reports and row-level power values used for the Capstone hierarchical power model construction in the paper cannot be released because they contain information protected by an NDA.
The artifact therefore separates two workflows.

- **Figures 5, 9, and 10 are functional demonstrations using synthetic data.** The reports under
  `data/power_model_synthetic/` do not contain values from the private PTPX reports. They exercise the same parsing, feature extraction, hierarchical nonnegative modeling, activity-proxy, oracle, and plotting workflow. The public synthetic fit uses architecture-informed event-to-row compatibility constraints to make the limited-data fitting problem identifiable. The synthetic data is intentionally not tuned to reproduce the paper's exact results.
- **The controller experiments use the released deployed coefficients.** The
  coefficients in `coeffs/` were learned from the private reports and are also embedded as defaults in `src/pipeline.py`. 
- A full ASIC flow and PTPX are not needed to run this public artifact.
  They are only needed to create a new signoff-labeled dataset, if desired.

## Clone the repository

```bash
git clone https://github.com/usc-acorn/yarzada-capstone-micro2026.git
cd yarzada-capstone-micro2026
```

## Quick Start: regenerate all outputs from the included data

This path uses only the pre-generated public data in `data/`. From the repository
root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 src/capstone_power_model.py evaluate \
  --manifest data/power_model_synthetic/manifest.json \
  --output-dir figures/generated_power_model \
  --overwrite \
  --validate-reference

python3 plot/generate_capstone_figures.py \
  --data-dir data \
  --output-dir figures \
  --validate-reference
```

The first command generates synthetic functional demonstrations corresponding
to Figures 5, 9, and 10 under `figures/generated_power_model/`. The second
generates the released paper/reference versions of Figures 5 and 9, along with
Figures 11–13 and Tables VII–IX, directly under `figures/`. A successful run
ends with these messages:

```text
REFERENCE VALIDATION: PASS — bundled synthetic power-model data matches the reference signature.
REFERENCE VALIDATION: PASS — bundled controller data matches the reference signature.
```

The checks compare the numerical inputs and derived metrics used by the
figures, and also confirm that all expected output files were written. View the generated figures in `figures/`.

### Expected behavior of synthetic power-model figures

Figures 5, 9, and 10 use deterministic synthetic PTPX data, so their
numerical values and per-kernel trends are not expected to match the
corresponding paper figures. A successful run should instead match the
bundled reference outputs and report `REFERENCE VALIDATION: PASS`.

See [Appendix: Expected behavior of synthetic Figures 5, 9, and 10](#expected-behavior-of-synthetic-figures-5-9-and-10)
for details on how each figure should be interpreted.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/pipeline.py` | AHA `pipeline.py` with simultaneous Baseline and Capstone I/II/III evaluation |
| `src/sta.py` | AHA STA entry point modified to call the RTL translator |
| `src/app_graph_to_RTL.py` | Writes `application_graph.v` from the routed application graph |
| `src/capstone_power_model.py` | Synthetic power-model fitting and Figures 5, 9, and 10 |
| `coeffs/` | Released deployed Capstone power model coefficients and Capstone III bounds |
| `plot/generate_figure12_sweep_data.py` | Derives the Figure 12 cap sweep from a complete tensor3-ttv trace |
| `plot/generate_capstone_figures.py` | Generates Figures 5, 9, 11–13 and Tables VII–IX |
| `data/capstone_<kernel>/` | Five saved files from one AHA all-modes run for a kernel |
| `data/power_model_synthetic/` | Synthetic hierarchy reports |
| `figures/` | Generated output figures |

## Requirements

The controller runs use the
[Stanford AHA Tutorial](https://stanfordaha.github.io/aha_tutorial/) Docker
image. The official
[Docker instructions](https://stanfordaha.github.io/aha_tutorial/docker/)
use `stanfordaha/garnet:micro-demos`.

Local figure generation requires Python 3 and the packages in
`requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 1. Generate controller data in the AHA Docker container

### 1.1 Start the container

Run these commands on the host machine.

```bash
docker pull stanfordaha/garnet:micro-demos
docker run -it -d --name capstone stanfordaha/garnet:micro-demos bash
```

If the container already exists, start it with:

```bash
docker start capstone
```

### 1.2 Install the Capstone files

From the root of this repository on the host:

```bash
docker cp src/pipeline.py \
  capstone:/aha/archipelago/archipelago/pipeline.py
docker cp src/sta.py \
  capstone:/aha/archipelago/archipelago/sta.py
docker cp src/app_graph_to_RTL.py \
  capstone:/aha/archipelago/archipelago/app_graph_to_RTL.py
docker cp coeffs \
  capstone:/aha/archipelago/archipelago/
```

Open a shell in the container:

```bash
docker exec -it capstone bash
```

Most modified files are now under:

```text
/aha/archipelago/archipelago
```

### 1.3 Choose a kernel

Inside the container, edit the `--sam_graph` argument in
`/aha/cascade_demo.sh`, and make sure the width and height are set to
`--width 32 --height 16`.

For example:

```text
--sam_graph /aha/sam/compiler/sam-outputs/onyx-dot/vec_elemadd.gv
```

The eight evaluated graph names are:

```text
vec_elemadd.gv
mat_elemmul.gv
tensor3_ttv.gv
tensor3_mttkrp.gv
tensor3_innerprod.gv
mat_sddmm.gv
mat_mask_tri.gv
mat_mattransmul.gv
```

### 1.4 Configure and run one kernel

Set the common options once inside the container:

```bash
export CAPSTONE_RUN_ALL_MODES=1
export CAPSTONE_MODEL_DIR=/aha/archipelago/archipelago
export CAPSTONE_BOUNDS_JSON=/aha/archipelago/archipelago/coeffs/capstone_iii_bounds.json
export CAPSTONE_REQUIRE_TABLE8_BOUNDS=1
export CAPSTONE_USE_FREQ_II_SCALING=1
export CAPSTONE_FREQ_REF_MHZ=100
export CAPSTONE_II_SCALE_BY_FREQ=1
export CAPSTONE_PRIMARY_OUTPUT_MODE=capstone_iii_full
export NUM_BITSTREAMS=4
unset CAPSTONE_II_CALIBRATION_JSON
```

Then set the kernel-specific values. For `vec_elemadd`:

```bash
export CAPSTONE_RUN_ID=vec_elemadd
export CAPSTONE_POWER_CAP_MW=350
export CAPSTONE_II_ANCHOR_Q_MW=22.4
export CAPSTONE_II_SPEC_Q_MW=14.0

cd /aha
./cascade_demo.sh max
```

`max` asks Cascade to continue breaking critical paths until it reaches maximal
pipelining. Capstone evaluates all controller modes on this shared trajectory.
When one mode crosses its cap, logging stops only for that mode. The other
modes and the uncapped baseline continue.

Each kernel uses a different power cap because its mapped design has a different power range, so the caps are selected relative to each kernel’s operating range to provide a meaningful and comparable power constraint. Additionally, Capstone II adds a safety margin to the predicted power before comparing a candidate against the cap. For these runs, the anchor and speculative margins are set to 6.4% and 4.0% of the kernel-specific power cap, respectively, with the larger anchor margin providing the more conservative selection. Use the following values for the eight runs:

| Kernel | Cap (mW) | Capstone II anchor margin (mW) | Capstone II speculative margin (mW) |
| --- | ---: | ---: | ---: |
| `vec_elemadd` | 350 | 22.4 | 14.0 |
| `mat_elemmul` | 650 | 41.6 | 26.0 |
| `tensor3_ttv` | 700 | 44.8 | 28.0 |
| `tensor3_mttkrp` | 1300 | 83.2 | 52.0 |
| `tensor3_innerprod` | 750 | 48.0 | 30.0 |
| `mat_sddmm` | 1300 | 83.2 | 52.0 |
| `mat_mask_tri` | 1100 | 70.4 | 44.0 |
| `mat_mattransmul` | 1000 | 64.0 | 40.0 |

Change `--sam_graph`, `CAPSTONE_RUN_ID`, the cap, and the two margins before
each run.

The run writes these files under `/aha/garnet/SIM_DIR/`:

```text
capstone_all_modes_trace.csv
capstone_all_modes_summary.csv
capstone_all_modes_bitstreams.csv
capstone_figure11_timing.csv
capstone_all_modes_selection.json
```

The final `sta.py` step also writes:

```text
/aha/garnet/SIM_DIR/application_graph.v
```

By default, `application_graph.v` represents the selected full-bounds
Capstone III result. Set `CAPSTONE_PRIMARY_OUTPUT_MODE` to another recorded mode
if a different final RTL is needed.

### 1.5 Copy each run back to the repository

Run this on the host immediately after each kernel because the next run reuses
`/aha/garnet/SIM_DIR`.

```bash
kernel=vec_elemadd
mkdir -p "data/capstone_${kernel}"

for file in \
  capstone_all_modes_trace.csv \
  capstone_all_modes_summary.csv \
  capstone_all_modes_bitstreams.csv \
  capstone_figure11_timing.csv \
  capstone_all_modes_selection.json
do
  docker cp \
    "capstone:/aha/garnet/SIM_DIR/${file}" \
    "data/capstone_${kernel}/${file}"
done
```

Change `kernel` to match each run.

## 2. Generate the Figure 12 sweep data

Figure 12 evaluates several caps on one complete tensor3-ttv candidate
trajectory. Generate that trajectory with a deliberately unreachable cap so
all modes remain active through maximal pipelining.

Inside the container, select `tensor3_ttv.gv` and run:

```bash
export CAPSTONE_RUN_ID=tensor3_ttv_high_cap
export CAPSTONE_POWER_CAP_MW=1000000000
export CAPSTONE_II_ANCHOR_Q_MW=64000000
export CAPSTONE_II_SPEC_Q_MW=40000000

cd /aha
./cascade_demo.sh max
```

Copy the five result files to `data/capstone_tensor3_ttv_high_cap/`. Then derive the single sweep CSV locally:

```bash
python3 plot/generate_figure12_sweep_data.py \
  --trace data/capstone_tensor3_ttv_high_cap/capstone_all_modes_trace.csv \
  --output data/figure12_sweep.csv \
  --run-id tensor3_ttv_high_cap
```

The sweep script reselects candidates offline for each target cap. The very
large run cap is used to preserve the complete trajectory. When a trace contains
results from multiple runs, `--run-id` selects the run used to construct the
sweep data.

## 3. Generate Figures 5, 9, and 11–13 and Tables VII–IX locally

From the repository root:

```bash
python3 plot/generate_capstone_figures.py \
  --data-dir data \
  --output-dir figures
```

This command produces:

- Figure 11 from `data/capstone_tensor3_ttv/capstone_figure11_timing.csv`
- Figure 12 from `data/figure12_sweep.csv`
- Figure 13 from all eight kernel directories, combining normalized frequency
  and slack-to-cap in one two-panel plot
- Table VII with aggregate controller success, normalized frequency, slack, and
  returned-set size
- Table VIII from the Unpruned row, the five Capstone III bound modes, and the
  `K=90`, `K=8`, and `K=4` selections
- Table IX from the tensor3-innerprod and mat-sddmm selections, in addition to the fixed
  prior-work rows

Figure 11 reports mean execution time per iteration. The Cascade and Capstone
post-PnR loop timings are measured on the machine running the artifact and are
therefore expected to vary across hardware. The **Signoff power** timing comes
from a separate PTPX flow protected under NDA, so the public artifact
does not rerun that flow. Instead, the generated Figure 11 uses the Signoff
power timing value reported in the paper.

## 4. Generate the synthetic power-model demonstration

To regenerate the deterministic synthetic reports:

```bash
python3 src/capstone_power_model.py make-synthetic \
  --output-dir data/power_model_synthetic \
  --seed 7 \
  --overwrite
```

Fit the model and generate Figures 5, 9, and 10:

```bash
python3 src/capstone_power_model.py evaluate \
  --manifest data/power_model_synthetic/manifest.json \
  --output-dir figures/generated_power_model \
  --overwrite
```

## Appendix: Reproduction details and reported quantities

### Metrics used by Figure 13 and Tables VII–VIII

Slack uses the selected candidate's "mean power" (before an upper bound is added):

```text
DeltaCap (%) = 100 * (power_cap - P_mean_mW) / power_cap
```

- Figure 13 reports per-kernel selected frequency normalized to the same
  kernel's uncapped baseline and the corresponding slack to the power cap.
- Table VII summarizes success rate, average normalized frequency, median slack,
  and returned-set size across kernel-cap pairs.
- Table VIII normalizes each Capstone III sensitivity setting to full bounds
  with `K=90`.

### Controller modes

- **Baseline** uses the Capstone I guardband implementation with an infinite
  cap. It never stops the shared search and therefore selects the maximally
  pipelined candidate.
- **Capstone I** applies the paper's guardbands
  `gamma_anchor=0.45` and `gamma_spec=0.30`.
- **Capstone II** normally uses a conformal residual envelope. See the note on margins below.
- **Capstone III** uses the event-level bounds in
  `coeffs/capstone_iii_bounds.json`.

The deployed power model is:

```text
Ptotal = gamma_eff * sum(beta[e] * event_count[e])
       + sum(theta[j] * leakage_feature[j])
```

`gamma_eff` combines the learned activity proxy with frequency/II scaling
relative to 100 MHz.

### Capstone II engineering margins

The available calibration set is too small to support a
finite-sample anchor level `alpha_anchor=0.005`, which requires approximately
199 calibration samples. The public controller demonstration therefore uses:

```text
q_anchor = 0.064 * power_cap
q_spec   = 0.040 * power_cap
rho(f)   = max(1, f / 100 MHz)
```

The controller adds `q * rho(f)` to the mean prediction. This rule is
used for every kernel only to demonstrate Capstone II stopping and candidate
selection for this artifact. These runs do **not** provide a split-conformal coverage guarantee.

### Capstone III release bounds

The released bound components are scaled from the deployed pre-gamma dynamic
coefficients:

```text
fit      = 5%
activity = 4%
PVT      = 3%
OOD      = 3%
leakage  = 0.1 mW
```

The full bound is the sum of these components. `coeffs/capstone_iii_bounds.json` records the exact
event-level values used by the included traces. Because the private residual
data cannot be released, the public artifact demonstrates the controller and
Table VIII construction but does not independently recalibrate or certify these
bounds from signoff samples.

### Power model oracle in Figure 9

The deployable model is fitted on samples marked `train` and predicts activity using the compiler-visible proxy. The Figure 9 oracle is intentionally optimistic. It fits the same model abstraction on all synthetic samples, including held-out samples, and uses each sample's fitted activity factor `gamma`. It measures the best in-sample behavior available within the chosen
event abstraction.

### Scalar Aggregate NNLS and Signoff oracle in Table VII

Scalar Aggregate NNLS learns power only from total report power and aggregate event counts. It does not use hierarchy-row supervision. In the evaluated controller data, it underpredicts enough that it never reaches the cap, so it does not stop or alter pipelining. It consequently selects the same maximally pipelined candidate as uncapped Cascade.

The Table VII **Signoff oracle** row is different from the Figure 9 power model oracle. It requires per-candidate signoff power, which is not publicly released. Its aggregate values are directly transferred from the paper's real results.

### Table IX

Table IX is a capability-oriented comparison under representative power caps.
The designs use different technology nodes, fabrics, workloads, and reporting
methods, so the table is not an apples-to-apples architectural ranking.

The RipTide, Snafu, UE-CGRA, and Plasticine rows are fixed values obtained
from their released results. The Cascade and Capstone rows are computed from:

```text
data/capstone_tensor3_innerprod/capstone_all_modes_selection.json
data/capstone_mat_sddmm/capstone_all_modes_selection.json
```

For each original operating point, the script reads `f_mhz` and `P_mean_mW`.
The optimistic 2x and 4x architectural-throttling columns divide **both**
frequency and power by 2 or 4:

```text
2x: frequency / 2, power / 2
4x: frequency / 4, power / 4
```

Slack and success are recomputed after throttling. This models ideal
even-duty-cycle run time throttling.

### Expected behavior of synthetic Figures 5, 9, and 10

The public artifact uses deterministic synthetic PTPX data for Figures 5, 9,
and 10 because the original hierarchy reports and row-level power values
cannot be released.

The public synthetic power model demonstration uses architecture-informed
event-to-row compatibility constraints during hierarchical fitting. The
available signoff-labeled data are relatively small, and many compiler-visible
events co-vary across mapped configurations, making a fully unconstrained
event-to-row allocation poorly identifiable. The compatibility constraints
therefore restrict each event to physically plausible power report groups,
while the nonnegative contribution magnitudes and event costs are learned from
the hierarchical power data. 

- **Figure 5:** Calibrating only on `vec_elemadd` should fit that workload very
  closely but may generalize poorly to other workloads. Calibrating across all
  workloads should provide more consistent accuracy across kernels. The exact
  per-kernel error ordering is specific to the synthetic dataset.

- **Figure 9:** The Capstone model should track the synthetic reference power,
  while the oracle provides a more accurate optimistic reference. Because the
  synthetic data is cleaner than the real PTPX data, the resulting errors are
  lower than those reported in the paper.

- **Figure 10:** Within the architecture-informed compatibility constraints,
  the learned nonnegative allocations and event costs should produce a power
  breakdown consistent with the expected hardware components. In
  Figure 10(b), the model and synthetic PTPX power breakdowns should match
  closely because the synthetic reports are generated from the same simplified
  event structure used by the demonstration.

The expected outputs from the bundled synthetic dataset are provided in
`figures/generated_power_model/`. These outputs, along with the
`REFERENCE VALIDATION: PASS` message, should be used to verify the synthetic
workflow rather than comparing Figures 5, 9, and 10 directly against the
paper.