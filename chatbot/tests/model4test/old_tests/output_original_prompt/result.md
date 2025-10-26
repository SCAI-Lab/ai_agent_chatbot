Model: llama3.1:8b
  Overall -> precision: 0.149, recall: 0.167, F1: 0.158 (TP: 36, FP: 206, FN: 179)
  Redundancy rate: 0.851
  Time -> total: 2360.00s, mean: 472.00s, std: 80.92s
  Per topic:
    basic_info: P=0.385, R=0.667, F1=0.488 (TP=10, FP=16, FN=5)
    interests: P=0.260, R=0.200, F1=0.226 (TP=20, FP=57, FN=80)
    mental_state: P=0.214, R=0.060, F1=0.094 (TP=6, FP=22, FN=94)
    motivation: P=0.000, R=0.000, F1=0.000 (TP=0, FP=12, FN=0)
    preferences: P=0.000, R=0.000, F1=0.000 (TP=0, FP=48, FN=0)
    schedule: P=0.000, R=0.000, F1=0.000 (TP=0, FP=28, FN=0)
    work: P=0.000, R=0.000, F1=0.000 (TP=0, FP=23, FN=0)
  Per session:
    1: P=0.226, R=0.233, F1=0.230 (TP=7, FP=24, FN=23)
    2: P=0.167, R=0.189, F1=0.177 (TP=7, FP=35, FN=30)
    3: P=0.135, R=0.152, F1=0.143 (TP=7, FP=45, FN=39)
    4: P=0.143, R=0.160, F1=0.151 (TP=8, FP=48, FN=42)
    5: P=0.115, R=0.135, F1=0.124 (TP=7, FP=54, FN=45)

Model: mistral:7b-instruct
  Overall -> precision: 0.121, recall: 0.056, F1: 0.076 (TP: 12, FP: 87, FN: 203)
  Redundancy rate: 0.879
  Time -> total: 1860.27s, mean: 372.05s, std: 80.58s
  Per topic:
    basic_info: P=0.455, R=0.333, F1=0.385 (TP=5, FP=6, FN=10)
    interests: P=0.000, R=0.000, F1=0.000 (TP=0, FP=37, FN=100)
    mental_state: P=0.233, R=0.070, F1=0.108 (TP=7, FP=23, FN=93)
    work: P=0.000, R=0.000, F1=0.000 (TP=0, FP=21, FN=0)
  Per session:
    1: P=0.200, R=0.033, F1=0.057 (TP=1, FP=4, FN=29)
    2: P=0.154, R=0.054, F1=0.080 (TP=2, FP=11, FN=35)
    3: P=0.100, R=0.043, F1=0.061 (TP=2, FP=18, FN=44)
    4: P=0.133, R=0.080, F1=0.100 (TP=4, FP=26, FN=46)
    5: P=0.097, R=0.058, F1=0.072 (TP=3, FP=28, FN=49)

Model: qwen2.5:7b-instruct
  Overall -> precision: 0.225, recall: 0.181, F1: 0.201 (TP: 39, FP: 134, FN: 176)
  Redundancy rate: 0.775
  Time -> total: 2131.56s, mean: 426.31s, std: 140.74s
  Per topic:
    basic_info: P=1.000, R=0.667, F1=0.800 (TP=10, FP=0, FN=5)
    interest: P=0.000, R=0.000, F1=0.000 (TP=0, FP=34, FN=0)
    interests: P=0.266, R=0.210, F1=0.235 (TP=21, FP=58, FN=79)
    mental_state: P=0.276, R=0.080, F1=0.124 (TP=8, FP=21, FN=92)
    work: P=0.000, R=0.000, F1=0.000 (TP=0, FP=21, FN=0)
  Per session:
    1: P=0.318, R=0.233, F1=0.269 (TP=7, FP=15, FN=23)
    2: P=0.200, R=0.162, F1=0.179 (TP=6, FP=24, FN=31)
    3: P=0.259, R=0.152, F1=0.192 (TP=7, FP=20, FN=39)
    4: P=0.227, R=0.200, F1=0.213 (TP=10, FP=34, FN=40)
    5: P=0.180, R=0.173, F1=0.176 (TP=9, FP=41, FN=43)

