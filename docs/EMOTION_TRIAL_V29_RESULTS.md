# Matched trial-context training v29

Both linear models completed without convergence warnings.1,792 training trials
and135SEED/225unseen-video FACED/475seen-video FACED validation trials come from
the exactv28 subject/video split. Both use all retained trial windows at inference,
the identical training-window scaler, and matched total sample-weight mass.
[Protocol](EMOTION_TRIAL_V29_PROTOCOL.md).

| Training loss | C | SEED trial balanced % | FACED unseen-video trial balanced % | FACED seen-video trial balanced % | Selection mean % |
|---|---:|---:|---:|---:|---:|
| Window | 10 | 51.85 | 41.67 | 41.89 | 46.76 |
| Trial mean | 0.1 | 56.30 | 41.67 | 44.78 | 48.98 |

The source selection improvement is from SEED; held-video FACED is unchanged.
Both source-selected models and the common scaler were hash-verified before
evaluation on640SEED-VII HNS trials from20subjects, each with10fixed windows.

| Training loss | External trial accuracy % | Subject-balanced % | Balanced95% CI |
|---|---:|---:|---:|
| Window | 47.03 | 46.88 | 43.06–50.69 |
| Trial mean | 41.25 | 45.56 | 40.42–50.83 |

Trial-minus-window-loss subject-balanced difference is−1.32percentage points,
paired subject bootstrap95% CI[−5.00,+2.29]. Directly training on averaged features
did not produce a clear external gain. All source and external results are
reported; no alternative target-selected C or aggregation rule is promoted.

The window-trained control also uses full trial context at evaluation, so its
46.88% cannot be described as improved3s online recognition. The trial-mean
model's ordinary accuracy and balanced accuracy differ because class frequencies
and recalls differ; neither metric should be omitted. SEED-VII is repeatedly
exposed, and this is exploratory known-task emotion classification. No LLM or
text understanding is tested. MI unchanged; the overarching goal remains open.

Artifacts under`artifacts/emotion_trial_v29/` retain every C's source scores,
model/scaler hashes, fixed split hash, external predictions and paired statistics.
Grouping tests verify recording separation, duplicate-window rejection and label
consistency. Runtime checks confirm split membership and identical regularization
weight mass for both training units.
