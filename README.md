# Wine Quality Prediction — ANN

> **Course:** Deep Learning – Jensen Vocational College
> **Dataset:** [Wine Quality – Red & White](https://www.kaggle.com/datasets/rajyellow46/wine-quality)

---

## Overview

Wineries and distributors rely on expert tasters to assess wine quality — a process
that is slow, expensive, and subjective. A single expert can only evaluate a limited
number of wines per day, and results vary between tasters.

This project builds an Artificial Neural Network trained on physicochemical
measurements that predicts whether a wine is **good (quality ≥ 6)** or
**poor (quality < 6)** — instantly and consistently, with no human taster required.

| Property | Value                                          |
|----------|------------------------------------------------|
| Dataset  | Wine Quality — red & white (UCI / Kaggle)      |
| Task     | Binary classification                          |
| Target   | `quality` → `1` (good ≥ 6) / `0` (poor < 6)  |
| Model    | ANN — TensorFlow / Keras                       |
| Extra    | Grid Search, Dropout, L2 regularization        |

---

## Dataset

| Property      | Value                                         |
|---------------|-----------------------------------------------|
| Source        | Kaggle / UCI Machine Learning Repository      |
| Rows          | 6 497 (1 599 red + 4 898 white)               |
| Features      | 11 physicochemical properties                 |
| Target        | `quality` → `1` (good ≥ 6) / `0` (poor < 6) |
| Class balance | ~67 % good, ~33 % poor                       |

### Getting the data

1. Go to the [Kaggle page](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
2. Download `winequality-red.csv` and `winequality-white.csv`
3. Place both files in the `data/raw/` folder

> The raw files are **not committed** (see `.gitignore`).  
> Run `python main.py` to automatically generate the processed dataset.

---

## Project Structure

```
wine-quality-ann/
├── data/
│   ├── raw/                          ← place Kaggle CSV files here (not committed)
│   └── processed/                    ← cleaned & preprocessed data (auto-generated)
│
├── notebooks/
│   ├── 01_EDA.ipynb                  ← Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb       ← Baseline ANN — 2 hidden layers
│   ├── 03_optimization.ipynb         ← Grid Search, Dropout, L2 regularization
│   └── 04_story.ipynb                ← Data story / presentation
│
├── src/
│   ├── __init__.py                   ← package definition
│   ├── data_preprocessing.py         ← loading, cleaning, scaling, splitting
│   ├── model.py                      ← ANN build functions
│   ├── train.py                      ← training logic
│   └── evaluate.py                   ← metrics, plots, saving results
│
├── models/                           ← saved models (.keras)
├── reports/
│   └── figures/                      ← exported plots (.png)
│
├── main.py                           ← runs the full pipeline end-to-end
├── requirements.txt                  ← dependencies
└── README.md                         ← this file
```

---

## Models

| Model           | Purpose                                           |
|-----------------|---------------------------------------------------|
| Baseline ANN    | Simple 2-layer network — reference point          |
| Optimized ANN   | Dropout + L2 regularization — reduces overfitting |
| Grid Search ANN | Systematic hyperparameter tuning — best model     |

---

## Technologies

| Package              | Usage                              |
|----------------------|------------------------------------|
| TensorFlow / Keras   | Build and train the ANN            |
| scikit-learn         | Preprocessing, metrics, GridSearch |
| pandas / numpy       | Data handling                      |
| matplotlib / seaborn | Visualizations                     |
| Jupyter              | Interactive notebooks              |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/wine-quality-ann.git
cd wine-quality-ann

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
source .venv/Scripts/activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data in data/raw/ (see above)

# 5. Run the full pipeline
python main.py
```

---