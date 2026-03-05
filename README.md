# Binary Minds – ML Challenge Submission

## Problem
The objective is to detect whether a device is operating normally or experiencing a fault condition using 47 numerical features

Class Labels:
0 → Normal  
1 → Faulty

## Approach
1. Load training data from TRAIN.csv
2. Separate features (F01–F47) and target (Class)
3. Split data into 80% training and 20% validation
4. Train a RandomForestClassifier with 200 trees
5. Achieved ~98% validation accuracy
6. Retrained model on full dataset
7. Generate predictions for TEST.csv

## Project Structure

```
binary-minds-ml/
│
├── data/
│   ├── TRAIN.csv
│   └── TEST.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── main.py
├── FINAL.csv
├── README.md
└── .gitignore
```

## How to Run

Install dependencies:

pip install pandas scikit-learn

Run the pipeline:

python main.py

## Output

Running the program generates:

FINAL.csv

Format:

ID,CLASS