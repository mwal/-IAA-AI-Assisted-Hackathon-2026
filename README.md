# Insurance pricing across developed and emerging markets

Code and presentation for the IAA AI Assisted Hackathon 2026. The solution predicts 2025 claim costs for Motor, Buildings and Contents and adapts developed-market risk models to an emerging market.

## Public repository scope

This repository publishes **source code, dependency requirements, instructions and slides only**. Raw competition data, saved model binaries, fitted preprocessing objects, reviewed row-level repair decisions reference prediction files, model specifications and verification records are excluded. The complete private reproducibility package remains local.

The inference and selected-refitting entry points depend on those private artifacts and specifications. **This public checkout alone cannot reproduce the selected submissions.** Refitting currently uses saved models as fixed-configuration templates; it is not a standalone train-from-scratch pipeline. Reviewers with authorized access must obtain the private artifacts, specifications and original competition data separately. Do not upload them without explicit permission.

## Approach and results

| Task | Selected model | Recorded public D² | Prediction rows |
|---|---|---:|---:|
| Developed market | Frequency–severity boosted models, plus a Buildings direct-cost blend | 0.12404 | 626,639 |
| Emerging market | Calibrated developed direct-cost transfer blended with reporting-aware frequency–severity predictions; gender calibration on the direct branch | 0.14543 | 209,063 |

Private scores for these selected files remain unknown. Public feedback influenced selection and is not an independent validation estimate.

Developed source models use 2021–2023 data. The final emerging calibration and local models use 2023–2024. Compatible predictors allow developed risk patterns to be transferred; overall levels are recalibrated to emerging experience. Selected direct-cost weights are 50% Motor, 50% Buildings and 100% Contents. The remaining share uses frequency–severity predictions.

The claims data provide one financial valuation. Loss and report dates support a reporting-delay adjustment to training/calibration exposure; forecasts still use full covered time. Incurred costs already include case reserves. Subsequent changes in reported costs cannot be measured directly without repeated valuations, so no measured financial-development uplift is claimed.

## AI-assisted workflow and validation

AI assisted with data audits, model implementation, diagnostics and independent prediction checks. Human review challenged conservative repair rules, interpreted categories, questioned information timing and directed the investigation of reporting maturity.

Historical tests use earlier years against later years and customer-separated emerging folds. New and continuing business are examined separately. The complete private package passed fresh-checkout prediction reconstruction and selected-model refitting checks. This does not imply that the public checkout was tested as a standalone runnable package. The model code is unchanged from the verified local version.

## Repository contents

- `reproduce.py`: raw-table assembly, saved-model inference, calibration, blends and comparison to private reference predictions.
- `train_selected.py`: refits the selected configurations using private saved-model templates; does not rerun historical searches.
- `office_premiums.py`: separate expense-based commercial premium build-ups.
- `src/`: data assembly, histories, feature design, preprocessing and reporting-delay calculations.
- `presentation/`: slides, narration and recording guide.

## Environment and authorized reproduction

The recorded environment is Python 3.9.6 on macOS. Exact dependency versions are in `requirements.txt`; fresh installation on another platform was not tested. The original competition data, model artifacts and specifications must be supplied separately. Both the flat developed layout with an `emerging/` folder and a `developed/` plus `emerging/` layout are supported.

Only after obtaining the authorized private `artifacts/` and `specifications/` folders and original data:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --data-dir /path/to/competition_data
```

Selected refitting also requires that private bundle:

```sh
python train_selected.py --data-dir /path/to/competition_data
python reproduce.py --data-dir /path/to/competition_data --artifacts-dir work/retrained_artifacts --skip-input-hashes --output-dir results/retrained
```

The complete private package reproduced all selected predictions within numerical precision in the recorded environment. Historical searches were not rerun. Seeds and selected settings were fixed. Financial snapshot limitations and portfolio shift remain material modeling assumptions.

## Video presentation

- [Six slides](presentation/Video_Submission_Demo.pptx)
- [Narration script](presentation/Narration_Submission_Demo.txt)
- [Recording guide](presentation/Recording_Guide_Submission_Demo.txt)

The presentation covers approach, results, AI-assisted workflow and changes when emerging-market data arrived. Slide 4 reproduces an actual historical count diagnostic; it is not ultimate-cost validation. The 520-word script targets about 4:50. Check actual recorded duration below five minutes. No recorded narration or MP4 is included.
