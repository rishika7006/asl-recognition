# Per-class precision / recall / F1 on the WLASL-10 test split

Note: the test split has only 13 samples spread across 8 of 10 classes; classes with `support=0` had no test examples and are reported as zeros for completeness.

### Farneback — SVM

```
              precision    recall  f1-score   support

        book     0.0000    0.0000    0.0000         0
       drink     0.1667    1.0000    0.2857         1
    computer     0.0000    0.0000    0.0000         1
      before     0.2500    0.3333    0.2857         3
          go     0.3333    1.0000    0.5000         1
       chair     0.0000    0.0000    0.0000         0
         who     0.0000    0.0000    0.0000         1
     clothes     0.0000    0.0000    0.0000         1
       candy     0.0000    0.0000    0.0000         3
      cousin     0.0000    0.0000    0.0000         2

    accuracy                         0.2308        13
   macro avg     0.0750    0.2333    0.1071        13
weighted avg     0.0962    0.2308    0.1264        13

```

### Farneback — MLP

```
              precision    recall  f1-score   support

        book     0.0000    0.0000    0.0000         0
       drink     0.5000    1.0000    0.6667         1
    computer     1.0000    1.0000    1.0000         1
      before     1.0000    0.6667    0.8000         3
          go     0.3333    1.0000    0.5000         1
       chair     0.0000    0.0000    0.0000         0
         who     0.5000    1.0000    0.6667         1
     clothes     0.0000    0.0000    0.0000         1
       candy     0.0000    0.0000    0.0000         3
      cousin     0.5000    0.5000    0.5000         2

    accuracy                         0.5385        13
   macro avg     0.3833    0.5167    0.4133        13
weighted avg     0.4872    0.5385    0.4795        13

```

### Raft — SVM

```
              precision    recall  f1-score   support

        book     0.0000    0.0000    0.0000         0
       drink     0.1429    1.0000    0.2500         1
    computer     0.0000    0.0000    0.0000         1
      before     0.5000    0.6667    0.5714         3
          go     0.5000    1.0000    0.6667         1
       chair     0.0000    0.0000    0.0000         0
         who     0.0000    0.0000    0.0000         1
     clothes     0.0000    0.0000    0.0000         1
       candy     0.0000    0.0000    0.0000         3
      cousin     0.0000    0.0000    0.0000         2

    accuracy                         0.3077        13
   macro avg     0.1143    0.2667    0.1488        13
weighted avg     0.1648    0.3077    0.2024        13

```

### Raft — MLP

```
              precision    recall  f1-score   support

        book     0.0000    0.0000    0.0000         0
       drink     0.0000    0.0000    0.0000         1
    computer     0.0000    0.0000    0.0000         1
      before     0.2500    0.3333    0.2857         3
          go     0.3333    1.0000    0.5000         1
       chair     0.0000    0.0000    0.0000         0
         who     1.0000    1.0000    1.0000         1
     clothes     0.0000    0.0000    0.0000         1
       candy     0.0000    0.0000    0.0000         3
      cousin     1.0000    0.5000    0.6667         2

    accuracy                         0.3077        13
   macro avg     0.2583    0.2833    0.2452        13
weighted avg     0.3141    0.3077    0.2839        13

```
