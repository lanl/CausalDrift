# Code Release for “Quantifying Temporal Drift Effects on Uncertainty, Interpretability, and Causal Fidelity” AML Summer Project

This repository contains the code for the AML summer project entitled “Quantifying Temporal Drift Effects on Uncertainty, Interpretability, and Causal Fidelity” at Los Alamos National Laboratory (LANL).  **The code was written by [Hoin Jung](https://hoinjung.github.io/).** The project focuses on benchmarking how changes in data over time affect the reliability of machine-learning models, with attention to uncertainty estimates, model explanations, and causal fidelity.

`CausalDrift` contains the public benchmark code for CausalDrift paper:

- generate synthetic and pseudo-realistic causal-drift data;
- run ten fixed causal-discovery baselines;
- evaluate every model with ten random fixed windows;
- make a trade-off plot and a μ−σ ranking plot.

Grid search and generated results are not included.

## Benchmark protocol

Each trajectory is evaluated on ten random windows. The window length is half
of the trajectory unless `--window-size` is supplied.
Each window is scored against the ground-truth adjacency at its last included
time step (`stop - 1`). The adjacency stores continuous directed-edge strength.
AUROC, AUPRC, and normalized SHD use its binary support (strength `> 1e-8`),
while normalized weighted Hamming distance uses the continuous strength. The
four reported metrics are:

| Metric | Direction |
| --- | --- |
| AUROC | higher is better |
| AUPRC | higher is better |
| normalized SHD | lower is better |
| normalized weighted Hamming distance | lower is better |

`summary_metrics.csv` reports population mean and standard deviation (`ddof=0`)
over the ten random windows for every dataset/model pair. `overall_metrics.csv`
averages those dataset-level mean/std values by model.

## Install

```bash
cd code_CausalDrift
pip install -e ".[classical,external]"
```

The base package provides generators, metrics, plots, and the in-tree NGC
implementation. To use the full default model set, clone the five optional
upstream sources listed in [third_party/README.md](third_party/README.md).
The runner reports an unavailable requested model as an error.

## Quick start

Generate a small synthetic smoke dataset, run two models, and make both plots:

```bash
causaldrift generate synthetic \
  --out data/synthetic-smoke \
  --instances 1 \
  --timesteps 200 \
  --drift-types edge \
  --temporal-patterns abrupt_appearance

causaldrift evaluate \
  --data data/synthetic-smoke \
  --out results/synthetic-smoke \
  --models pcmci+ ngc

causaldrift plot \
  --results results/synthetic-smoke \
  --out results/synthetic-smoke/figures
```

The default evaluator always uses `--num-windows 10` and random fixed windows.
No grid search occurs in any command.

## Generate data

The synthetic generator retains eight drift mechanisms:
`edge`, `mechanism`, `strength`, `node`, `driver`, `confounder`, `lag`, and
`noise`, each with four abrupt/gradual patterns. The default command writes ten
replicates of all requested cases.

`mechanism`, `driver`, `confounder`, and `noise` change the observations without
changing observed directed-edge support. For `confounder`, the two targets of
the latent common cause are stored in the metadata.

```bash
causaldrift generate synthetic --out data/synthetic
causaldrift generate pseudo --out data/pseudo
```

Pseudo-realistic data includes an ENSO-inspired coupled climate-index system,
SEIR, and SEIHRDV.

## Run all ten models

```bash
causaldrift evaluate \
  --data data/pseudo \
  --out results/pseudo \
  --models all \
  --third-party-dir third_party
```

The model set is fixed and can be inspected with `causaldrift models`.

| ID | Model | Fixed configuration |
| --- | --- | --- |
| `acd` | ACD | amortized, independently fitted per window, 100 epochs |
| `cuts` | CUTS | 100 epochs |
| `grasp` | GRaSP | BIC-covariance, depth 3 |
| `ngc` | NGC | in-tree LSTM, 100 iterations |
| `uncle` | UnCLe | amortized, independently fitted per window, 50 reconstruction + 100 joint epochs |
| `varlingam` | VARLiNGAM | lag 1 |
| `cdans` | CDANs | Fisher-Z, `tau_max=2` |
| `fpcmci` | F-PCMCI | correlation selector, lag 1 |
| `kausal` | Kausal | MLP [8,16], 30 bootstraps, one epoch |
| `pcmci+` | PCMCI+ | partial correlation, lag 1 |

Outputs are saved under `predictions/`, `metrics.csv`, `summary_metrics.csv`,
and `overall_metrics.csv`.

## Figures

`causaldrift plot` writes:

- `tradeoff.png`: mean performance versus random-window standard deviation;
  lower variability lies farther right.
- `ranking_mu_minus_sigma.png`: each metric's higher-is-better μ−σ rank and a
  balanced four-metric μ−σ rank.

## Publication checklist

Add `LICENSE` and `CITATION.cff` before release. Review the external model
licenses in `third_party/README.md`.

## AML Summer School

This project was developed as part of LANL’s AML summer program. The [AML fellowship website](https://www.lanl.gov/engage/collaboration/internships/summer-schools/applied-machine-learning-fellowship) describes the program, currently titled the **Advancing Machine Learning for Scientific Discovery Fellowship**, as an intensive 10-week summer research experience for graduate and upper-level undergraduate students across science, mathematics, computer science, and technology.

Participants conduct hands-on machine-learning research with mentors who bring scientific, computational, and machine-learning expertise. The program includes work on high-performance computing clusters, seminars from LANL researchers and external speakers, and opportunities to communicate research through discussions and presentations. Its summer projects address scientific research questions and aim to support co-authored, peer-reviewed publications. Further information is available on the [LANL AML program page](https://www.lanl.gov/engage/collaboration/internships/summer-schools/applied-machine-learning-fellowship).

## Release

The LANL open-source release identifier for this repository is **O5162**. This identifier is included to support confirmation of the copyright assertion for this software.

## Copyright

© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

## License

This code repository is distributed under the BSD-3-Clause License:

Copyright 2026. Triad National Security, LLC.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

