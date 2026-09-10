## Table of contents
* [General info](#general-info)
* [Building](#building)
* [Quick start: base gVAMP](#quick-start-base-gvamp)
* [Running TL-gVAMP (transfer learning)](#running-tl-gvamp-transfer-learning)
  * [Overview](#overview)
  * [Step 1 — base run per source population](#step-1--base-run-per-source-population)
  * [Step 2 — TL run on the target population](#step-2--tl-run-on-the-target-population)
  * [Step 3 — evaluate on held-out data](#step-3--evaluate-on-held-out-data)
  * [Multiple source populations](#multiple-source-populations)
  * [MAF-aware source weighting (optional)](#maf-aware-source-weighting-optional)
* [Convergence/stability options](#convergencestability-options)
* [Full option reference](#full-option-reference)

## General info

This repository implements Vector Approximate Message Passing (VAMP) for inference in
Genome-Wide Association Studies (GWAS). It supports:

- a **base** run of gVAMP on a single cohort ("population"), and
- a **transfer-learning (TL) run**, where inference on a target population is informed by
  effect-size estimates already computed for one or more source populations, via
  `--use-tl-lmmse`.

## Building

```bash
module purge
module load gcc openmpi boost

SRC=<path/to/gVAMP/src>
BIN=<path/to/gVAMP/bin>

mpic++ ${SRC}/main_real.cpp ${SRC}/vamp.cpp ${SRC}/utilities.cpp ${SRC}/data.cpp ${SRC}/options.cpp \
    -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG \
    -o ${BIN}/main_real.exe
```

This produces a single executable, `main_real.exe`, used for every mode below
(`--run-mode infere`, `test`, etc.) — the mode is selected at runtime via `--run-mode`,
not by building a different binary.

## Quick start: base gVAMP

Runs plain VAMP inference on one population/cohort — no transfer learning involved.

```bash
mpirun -np <num_mpi_workers> ${BIN}/main_real.exe \
    --model linear \
    --run-mode infere \
    --bim-file <path/to/pop.bim> \
    --bed-file <path/to/pop.bed> \
    --phen-files <path/to/pop.phen> \
    --N <num_individuals> \
    --Mt <num_markers> \
    --out-dir <path/to/output_dir>/ \
    --out-name <run_name> \
    --iterations <max_iterations> \
    --num-mix-comp 5 \
    --probs 0.99,0.005,0.0025,0.00125,0.00125 \
    --vars 0,0.0000001,0.000001,0.00001,0.0001 \
    --learn-vars 1 \
    --seed 1 \
    --rho 0.2 \
    --gamma-damp 0.3
```

This writes, for every iteration `it` (1..`--iterations`), effect-size estimates to
`<out_dir>/<run_name>_it_<it>.bin` (and the matching `_r1_it_<it>.bin` message vector,
needed as an input if this population is later used as a TL *source* — see below).

To evaluate a saved estimate against held-out genotypes/phenotype:

```bash
mpirun -np 1 ${BIN}/main_real.exe \
    --model linear \
    --run-mode test \
    --bed-file-test <path/to/pop_val_or_test.bed> \
    --phen-files-test <path/to/pop_val_or_test.phen> \
    --N-test <num_individuals_in_val_or_test> \
    --Mt-test <num_markers> \
    --out-dir <path/to/output_dir>/ \
    --out-name <run_name>_val \
    --estimate-file <path/to/output_dir>/<run_name>_it_1.bin \
    --test-iter-range 1,<max_iterations>
```

`--estimate-file` only needs to name iteration 1 — with `--test-iter-range` set, the
program substitutes each iteration number in turn (`_it_1.bin`, `_it_2.bin`, ...) and
prints one R² per iteration, plus the best (`max R2` / `max ind`). Use the iteration with
the best **validation** R² (not training R²) as the population's "best iteration" for
everything downstream — training R² keeps improving even after the model has started
overfitting, so it's not a reliable stopping criterion on its own.

## Running TL-gVAMP (transfer learning)

### Overview

TL-gVAMP never pools individual-level genotypes across populations. Instead, each source
population is first run through a normal base gVAMP inference, producing a message vector
(`r1`) and a scalar precision (`gam1`) at its best iteration. The target population's
inference then folds in those source summaries — never the source's raw genotypes or
phenotypes — as an extra LMMSE term.

The three steps below always run in this order: base run(s) on the source population(s),
then the TL run on the target, then evaluation.

### Step 1 — base run per source population

Run the [base gVAMP quick start](#quick-start-base-gvamp) above for every population that
will act as a *source*, then also run and validate it (`--run-mode test` on that
population's own held-out validation set) to find its best iteration. From that iteration
you need two things for step 2:

- the `r1` message vector: `<out_dir>/<source_run_name>_r1_it_<best_iter>.bin`
- the matching `gam1` value, scaled by that source population's sample size `N_source`:
  `gam1_scaled = gam1_at_best_iter * N_source`. (`gam1` at each iteration is stored
  alongside R² in the inference log; parsing it into a value you can pass on the command
  line is a one-line lookup, not a separate run.)

### Step 2 — TL run on the target population

```bash
mpirun -np <num_mpi_workers> ${BIN}/main_real.exe \
    --model linear \
    --run-mode infere \
    --bim-file <path/to/target.bim> \
    --bed-file <path/to/target.bed> \
    --phen-files <path/to/target.phen> \
    --N <num_individuals_target> \
    --Mt <num_markers> \
    --out-dir <path/to/tl_output_dir>/ \
    --out-name <tl_run_name> \
    --iterations <max_iterations> \
    --num-mix-comp 5 \
    --probs 0.943357346,0.028321334,0.014160667,0.007080333,0.003540167 \
    --vars 0,0.0000001,0.000001,0.00001,0.0001 \
    --learn-vars 1 \
    --seed 1 \
    --rho 0.05 \
    --gamma-damp 0.3 \
    --use-tl-lmmse 1 \
    --gamma-tl <gam1_scaled_from_source> \
    --r-tl-file <path/to/source_run_name>_r1_it_<best_iter>.bin
```

The only difference from a base run is the last three options: `--use-tl-lmmse 1` turns
on the TL term, `--gamma-tl` sets its precision, and `--r-tl-file` points at the source's
saved `r1` vector. Everything else (`--probs`, `--vars`, `--rho`, `--seed`, ...) is
typically its own separate config from the base run's, since the TL run tends to need
different tuning (e.g. a smaller `--rho`).

### Step 3 — evaluate on held-out data

Identical to the base-run evaluation above, just pointed at the TL output directory and
run name:

```bash
mpirun -np 1 ${BIN}/main_real.exe \
    --model linear \
    --run-mode test \
    --bed-file-test <path/to/target_val_or_test.bed> \
    --phen-files-test <path/to/target_val_or_test.phen> \
    --N-test <num_individuals_in_val_or_test> \
    --Mt-test <num_markers> \
    --out-dir <path/to/tl_output_dir>/ \
    --out-name <tl_run_name>_val \
    --estimate-file <path/to/tl_output_dir>/<tl_run_name>_it_1.bin \
    --test-iter-range 1,<max_iterations>
```

### Multiple source populations

`--gamma-tl`, `--r-tl-file`, and (if using MAF weighting) `--maf-pop2-file` each accept a
**comma-separated list**, one entry per source, all in the same order:

```bash
    --use-tl-lmmse 1 \
    --gamma-tl <gam1_scaled_source_A>,<gam1_scaled_source_B>,<gam1_scaled_source_C> \
    --r-tl-file <path/to/source_A>_r1_it_<best_iter_A>.bin,<path/to/source_B>_r1_it_<best_iter_B>.bin,<path/to/source_C>_r1_it_<best_iter_C>.bin
```

Each source's own precision (`gamma-tl`) is re-estimated adaptively during the TL run, so
the value passed on the command line only sets its starting point.

### MAF-aware source weighting (optional)

To down-weight a source population's per-marker contribution by how much its allele
frequency differs from the target's, add a target `.frq` file and one source `.frq` file
per source (same order as `--r-tl-file`):

```bash
    --maf-pop1-file <path/to/target.frq> \
    --maf-pop2-file <path/to/source_A.frq>,<path/to/source_B.frq>,<path/to/source_C.frq>
```

`.frq` files are the standard PLINK allele-frequency report (`plink --freq`).

## Convergence/stability options

Two options exist specifically to keep the iterative updates from oscillating or
diverging on noisy real-data cohorts — worth knowing about if you see R² degrade or go
negative partway through a run:

| Option | What it damps |
| --- | --- |
| `--gamma-damp <0,1]` | Blends each new `gam1`/`gam2`/`gamw` precision update with its previous value (`new = λ·candidate + (1−λ)·old`) instead of overwriting it outright. `1.0` = no damping (the original update rule); lower values damp harder. `0.3` is a reasonable starting point for large, real (non-simulated) marker panels. |
| `--use-lmmse-damp <0/1>` | Additionally damps the LMMSE-side message vectors themselves (`r2`/`x2_hat`), not just their precisions. Off (`0`) by default; consider turning on if `--gamma-damp` alone isn't enough. |

The prior (mixture-of-Gaussians) update is itself capped so its implied heritability can
never exceed what the estimated noise level (`gamw`) allows — this happens automatically
and needs no flag, but if you see a `[prior rescale] h2 ... -> ...` line in the log, it
means the expectation-maximization step tried to claim more heritability than is
physically possible and was corrected. Seeing this once or twice early in a run is normal;
seeing it on every iteration throughout a run signals the prior is persistently
misfitting and is worth a closer look.

## Full option reference

### Input data

| Option | Description |
| --- | --- |
| `--bim-file` | path to `.bim` file (training set) |
| `--bed-file` | path to `.bed` file (training set) |
| `--phen-files` | path to phenotype file (training set; only one phenotype supported) |
| `--cov-file` | path to `.cov` covariate file (probit model only) |
| `--bed-file-test` | path to `.bed` file (held-out set, used with `--run-mode test`) |
| `--phen-files-test` | path to phenotype file (held-out set) |
| `--N` | number of individuals in the training set |
| `--Mt` | total number of markers in the training set |
| `--N-test` | number of individuals in the held-out set |
| `--Mt-test` | total number of markers in the held-out set |
| `--true-signal-files` | path to a file with the true per-marker signal (simulations only) |
| `--C` | number of covariates (probit model only) |

### Output

| Option | Description |
| --- | --- |
| `--out-dir` | output directory (trailing slash recommended) |
| `--out-name` | prefix used for all files written by this run |
| `--store-pvals` | whether to also store per-marker association p-values |

### Run mode

| Option | Description |
| --- | --- |
| `--model` | `linear` or `bin_class` |
| `--run-mode` | `infere` (train), `test` (evaluate a saved estimate on held-out data), `both`, `restart`, `predict`, `predict_single` |
| `--estimate-file` | path to a saved `_it_<n>.bin` estimate (used by `--run-mode test`/`restart`/`predict*`) |
| `--test-iter-range` | `min,max` — range of saved iterations to evaluate under `--run-mode test` |
| `--init-est` | whether to initialize this run from `--estimate-file` instead of a cold start |

### Core VAMP settings

| Option | Description |
| --- | --- |
| `--iterations` | maximum number of outer VAMP iterations |
| `--seed` | RNG seed (non-negative integer) |
| `--rho` | damping factor applied to the Onsager-correction term (`alpha1`) |
| `--gamma-damp` | damping factor for the `gam1`/`gam2`/`gamw` precision updates — see [Convergence/stability options](#convergencestability-options) |
| `--use-lmmse-damp` | whether to also damp the LMMSE message vectors — see [Convergence/stability options](#convergencestability-options) |
| `--CG-max-iter` | max iterations of the conjugate-gradient solver used in the LMMSE step |
| `--stop-criteria-thr` | relative-error threshold for early stopping |
| `--alpha-scale` | genotype columns are scaled by `std^(-alpha-scale)` |
| `--use-XXT-denoiser` | use the `N x N`-inversion form of the denoiser instead of the default |
| `--gam1-init` | initial `gam1` (used with `--run-mode restart`) |
| `--gamw-init` | initial noise precision `gamw` (used with `--run-mode restart`) |

### Prior (mixture-of-Gaussians)

| Option | Description |
| --- | --- |
| `--num-mix-comp` | number of mixture components, including the zero-spike |
| `--probs` | initial mixture weights, comma-separated, must sum to 1 |
| `--vars` | initial mixture variances, comma-separated (first entry is the zero-spike, always `0`) |
| `--learn-vars` | whether mixture variances are re-estimated (`1`) or held fixed (`0`) |
| `--EM-err-thr` | relative-error convergence threshold for the prior's inner EM loop |
| `--EM-max-iter` | max iterations of the prior's inner EM loop |

### Transfer learning (TL-gVAMP)

See [Running TL-gVAMP](#running-tl-gvamp-transfer-learning) for the full workflow.

| Option | Description |
| --- | --- |
| `--use-tl-lmmse` | `1` to enable the TL term in the LMMSE step, `0` for a plain base run |
| `--gamma-tl` | starting precision for each TL source; comma-separated, one value per source, same order as `--r-tl-file` |
| `--r-tl-file` | path to each source's saved `r1` message vector; comma-separated, one per source |
| `--maf-pop1-file` | target population's `.frq` allele-frequency file (optional, enables MAF-aware source weighting) |
| `--maf-pop2-file` | each source population's `.frq` file; comma-separated, one per source (used together with `--maf-pop1-file`) |
| `--r1-add-info-file` | (legacy single-source TL) path to an `r1` vector |
| `--gam1-add-info` | (legacy single-source TL) `gam1` value paired with `--r1-add-info-file` |
| `--a_scale` / `--a-scale-start-iter` | scales the likelihood contribution of source information from a given iteration onward |

### Freezing / simulation / misc

| Option | Description |
| --- | --- |
| `--use-freeze` | whether to freeze certain marker positions during inference |
| `--freeze-index-file` | file with a 0/1 flag per marker indicating which positions are frozen |
| `--cov-estimate-file` | saved covariate-effect estimate (probit model) |
| `--probit-var` | residual variance used in the probit model |
| `--h2` | heritability used to generate simulated phenotypes |
| `--CV` | number of causal variants used to generate simulated phenotypes |
| `--scheduler` | internal MPI work-scheduling option |
