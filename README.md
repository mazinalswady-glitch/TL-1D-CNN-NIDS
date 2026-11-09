# Reconstructed cross-domain intrusion-detection study

This archive contains the executable reconstruction, raw per-seed metrics and confusion matrices, aggregate CSVs and the generated figure. It implements the 29-feature experiment family from the manuscript with a documented 12-epoch cap. It supersedes the numerical claims in the unavailable original experiment package. Original checkpoints and omitted alternative models or novelty scores were not used.

## Public inputs

Use these three CSVs. The result files record SHA-256 hashes.

- ToN-IoT network CSV: https://raw.githubusercontent.com/thawatchai2799/SecurityArchitectureSmartCityIoT/master/data/TON_IoT_Train_Test_Network.csv
- UNSW-NB15 training CSV: https://raw.githubusercontent.com/tylerhoward15/tensorFlow-MLP-UNSW-NB15/master/UNSW_NB15_training-set.csv
- IDS-ML CIC-IDS2017 sample: https://raw.githubusercontent.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/HEAD/data/CICIDS2017_sample.csv

## Reproduction

Install requirements.txt and matplotlib 3.10.8. Run the following from this directory, passing local paths to the three downloaded files.

```sh
python run.py ton.csv training.csv results.json
python extended/experiments.py ton.csv training.csv extended/reverse.json --mode reverse
python extended/experiments.py ton.csv training.csv extended/labels.json --mode labels
python extended/experiments.py ton.csv training.csv extended/open.json --mode open
python extended/cic.py ton.csv training.csv cic.csv extended/cic.json
python extended/leakage.py ton.csv training.csv extended/leakage.json
python extended/aggregate.py
```

The default CNN protocol uses seeds 0–4 and no more than 12 epochs. Log scaling and normalisation are fitted on training predictors. The full labelled validation partition remains available for early stopping in reduced-training-label runs. The open-set JSON reports AUROC, average precision, test composition, thresholds, and recall and realised false alarms for nominal validation FPRs of 1%, 5% and 10%. The leakage-control JSON includes both the training-resampling comparison and the intentionally contaminated condition. These diagnostic scores are not clean test estimates.

The primary, reversed and two CIC transfer pairs are retrained on labelled target data. Reported CPU times can vary across machines. The code does not reproduce the old manuscript's 12-model comparison, layerwise fine-tuning, energy or Mahalanobis novelty scores. The regenerated manuscript restricts its claims to the executed comparisons.

## Archived release

Run `python verify_release.py` before regenerating any outputs to verify the archived copies. The hash manifest is for the supplied release outputs. Generated fitting times and numerical results can vary with hardware and software.

The citation metadata uses a provisional author name. Confirm the complete author list and spelling, and choose a code licence with the rights holders before making the repository public. Raw benchmark datasets are excluded. Refer to the original data providers for access and terms.
