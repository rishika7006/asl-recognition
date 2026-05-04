# Lighting robustness — good vs low light

**Methodological caveat.** The user_videos clips were included in the training set when the models were retrained, so the accuracy numbers below are train-time recall (inflated). What's still informative is the *gap* between good-light and low-light recall — a small gap means the model handles both lighting regimes consistently; a large gap means lighting shifts the feature distribution. The cosine-similarity table at the bottom is a classifier-independent robustness measure: how close are the flow features for the same sign filmed under the two lighting conditions?


## Farneback — SVM

| class | good-light | low-light | gap (good - low) |
|---|---|---|---|
| before | 1/1 (100%) | 1/1 (100%) | +0% |
| book | 0/1 (0%) | 0/1 (0%) | +0% |
| candy | 1/1 (100%) | 1/1 (100%) | +0% |
| chair | 0/1 (0%) | 1/1 (100%) | -100% |
| clothes | 0/1 (0%) | 0/1 (0%) | +0% |
| computer | 1/1 (100%) | 1/1 (100%) | +0% |
| cousin | 1/1 (100%) | 1/1 (100%) | +0% |
| drink | 1/1 (100%) | 1/1 (100%) | +0% |
| go | 1/1 (100%) | 1/1 (100%) | +0% |
| who | 1/1 (100%) | 1/1 (100%) | +0% |
| **TOTAL** | **7/10 (70%)** | **8/10 (80%)** | **-10%** |


## Farneback — MLP

| class | good-light | low-light | gap (good - low) |
|---|---|---|---|
| before | 1/1 (100%) | 1/1 (100%) | +0% |
| book | 1/1 (100%) | 1/1 (100%) | +0% |
| candy | 1/1 (100%) | 1/1 (100%) | +0% |
| chair | 1/1 (100%) | 1/1 (100%) | +0% |
| clothes | 1/1 (100%) | 1/1 (100%) | +0% |
| computer | 1/1 (100%) | 1/1 (100%) | +0% |
| cousin | 1/1 (100%) | 1/1 (100%) | +0% |
| drink | 1/1 (100%) | 1/1 (100%) | +0% |
| go | 1/1 (100%) | 1/1 (100%) | +0% |
| who | 1/1 (100%) | 1/1 (100%) | +0% |
| **TOTAL** | **10/10 (100%)** | **10/10 (100%)** | **+0%** |


## Raft — SVM

| class | good-light | low-light | gap (good - low) |
|---|---|---|---|
| before | 1/1 (100%) | 1/1 (100%) | +0% |
| book | 1/1 (100%) | 0/1 (0%) | +100% |
| candy | 0/1 (0%) | 1/1 (100%) | -100% |
| chair | 1/1 (100%) | 1/1 (100%) | +0% |
| clothes | 0/1 (0%) | 0/1 (0%) | +0% |
| computer | 1/1 (100%) | 1/1 (100%) | +0% |
| cousin | 1/1 (100%) | 1/1 (100%) | +0% |
| drink | 1/1 (100%) | 1/1 (100%) | +0% |
| go | 1/1 (100%) | 1/1 (100%) | +0% |
| who | 1/1 (100%) | 1/1 (100%) | +0% |
| **TOTAL** | **8/10 (80%)** | **8/10 (80%)** | **+0%** |


## Raft — MLP

| class | good-light | low-light | gap (good - low) |
|---|---|---|---|
| before | 1/1 (100%) | 1/1 (100%) | +0% |
| book | 1/1 (100%) | 0/1 (0%) | +100% |
| candy | 1/1 (100%) | 1/1 (100%) | +0% |
| chair | 1/1 (100%) | 1/1 (100%) | +0% |
| clothes | 0/1 (0%) | 0/1 (0%) | +0% |
| computer | 1/1 (100%) | 1/1 (100%) | +0% |
| cousin | 1/1 (100%) | 1/1 (100%) | +0% |
| drink | 1/1 (100%) | 1/1 (100%) | +0% |
| go | 1/1 (100%) | 1/1 (100%) | +0% |
| who | 1/1 (100%) | 1/1 (100%) | +0% |
| **TOTAL** | **9/10 (90%)** | **8/10 (80%)** | **+10%** |


## Feature similarity (cosine) between good-light and low-light clips of the same sign

Classifier-independent. Higher = more lighting-robust feature representation. Range -1..1; >0.9 is very stable, <0.5 is sensitive.

| class | Farnebäck | RAFT |
|---|---|---|
| before | -0.429 | -0.021 |
| book | +0.144 | +0.053 |
| candy | +0.005 | +0.103 |
| chair | -0.006 | +0.033 |
| clothes | +0.062 | +0.027 |
| computer | +0.118 | +0.071 |
| cousin | -0.109 | -0.128 |
| drink | +0.430 | +0.340 |
| go | -0.028 | -0.011 |
| who | +0.497 | +0.556 |
| **mean** | **+0.068** | **+0.102** |
