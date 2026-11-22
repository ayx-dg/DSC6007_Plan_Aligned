# Completed experiment results

Timestamp-boundary 70/15/15. Adaptive bandits: warm up before test, then matched-only updates during test. Frozen LR trained on development only. No Yahoo IPS/DR.

## Held-out replay

|Policy|Matched|Clicks|Replay CTR|
|---|---:|---:|---:|
|LinUCB (disjoint)|780|30|3.846%|
|LinUCB (hybrid)|788|30|3.807%|
|contextual greedy|732|29|3.962%|
|ε-greedy (context-free)|753|22|2.922%|
|Window LinUCB|804|25|3.109%|
|Random|777|28|3.604%|
|Logistic regression|771|25|3.243%|

## Thirty algorithm seeds on the fixed log
- Random: mean 2.692%, SD 0.618 pp
- contextual greedy: mean 2.993%, SD 0.556 pp
- ε-greedy (context-free): mean 2.886%, SD 0.622 pp

## Logistic regression
{
  "C": 10.0,
  "threshold_validation_f1": 0.060828039275371906,
  "grid": [
    {
      "C": 0.001,
      "validation_average_precision": 0.05895101627700229
    },
    {
      "C": 0.01,
      "validation_average_precision": 0.06243967679146868
    },
    {
      "C": 0.1,
      "validation_average_precision": 0.06368417013529784
    },
    {
      "C": 1.0,
      "validation_average_precision": 0.0639283967392041
    },
    {
      "C": 10.0,
      "validation_average_precision": 0.06397850050398367
    }
  ],
  "validation": {
    "average_precision": 0.06397850050398367,
    "roc_auc": 0.7353907706141063,
    "log_loss": 0.108388460253303,
    "brier": 0.023985464256100278,
    "precision": 0.07631874298540965,
    "recall": 0.3008849557522124,
    "f1": 0.12175470008952552,
    "confusion_matrix": [
      [
        15912,
        1646
      ],
      [
        316,
        136
      ]
    ]
  },
  "test": {
    "average_precision": 0.07206485517318366,
    "roc_auc": 0.7373424164667819,
    "log_loss": 0.11318949647690911,
    "brier": 0.025313670993600438,
    "precision": 0.0861952861952862,
    "recall": 0.3047619047619048,
    "f1": 0.13438320209973753,
    "confusion_matrix": [
      [
        14009,
        1357
      ],
      [
        292,
        128
      ]
    ]
  },
  "test_click_prevalence": 0.0266058532877233,
  "features": "user6 + article6 + outer36; no history leakage",
  "test_protocol": "frozen training-only model; validation selects C and threshold"
}

## Synthetic known-propensity OPE
{
  "source": "synthetic randomized logger only; fixed target; q=0.5 fixed independently",
  "known_propensity": 0.25,
  "oracle_mean": 0.538374212077636,
  "IPS": 0.5356,
  "SNIPS": 0.5392670157068062,
  "DR": 0.539
}

## Interpretation limits
Replay is descriptive until logger/preprocessing assumptions are verified. Different policies accept different events. Random-seed variation is not population uncertainty. Synthetic drift does not establish real-user improvement. No unobserved candidate labels or test thresholds were used. Average precision (AP) is reported rather than trapezoidal PR-AUC. Simulation settings were fixed, not optimized on Yahoo test results.