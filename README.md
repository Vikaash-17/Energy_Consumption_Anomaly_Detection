# Energy Consumption Anomaly Detection

## AI-Based Energy Monitoring System

A machine learning system for detecting abnormal energy consumption patterns across commercial buildings using **Random Forest classification**, temporal feature engineering, building-specific normalization, and classification-threshold optimization.

The **source code and sample dataset** are maintained in this GitHub repository, while the trained model artifacts are hosted separately on Hugging Face.

**Trained Model:**
[Vikaash17/Energy-Consumption-Anomaly-Detection](https://huggingface.co/Vikaash17/Energy-Consumption-Anomaly-Detection)

---

## Project Overview

Energy consumption anomaly detection is a critical challenge for organizations managing large-scale building infrastructure. Monitoring hundreds of smart meters simultaneously makes manual anomaly identification time-consuming and error-prone.

This project automatically identifies abnormal energy consumption patterns from historical meter readings. The system processes building-level energy data, generates temporal and statistical features, normalizes consumption using building-specific scalers, and uses a trained Random Forest classifier to classify readings as:

* **Normal (0)**
* **Anomaly (1)**

The project was developed using approximately **200 commercial buildings** with hourly energy consumption data.

---

## Objectives

The main objectives of this project are:

* Preprocess raw energy meter readings.
* Extract meaningful temporal and statistical features.
* Normalize energy consumption separately for each building.
* Train a machine learning model for anomaly detection.
* Handle the highly imbalanced anomaly class.
* Optimize the classification threshold for improved anomaly recall.
* Evaluate model performance globally and across individual buildings.
* Provide a reusable prediction pipeline.
* Host trained model artifacts separately from the source-code repository.

---

## Dataset

The dataset consists of hourly energy meter readings collected from commercial buildings over approximately one year.

### Dataset Statistics

| Attribute           | Details                |
| ------------------- | ---------------------- |
| Number of buildings | 200                    |
| Reading frequency   | Hourly                 |
| Time period         | Approximately one year |
| Raw features        | 3                      |
| Engineered features | 15                     |
| Normal class        | ~98.5%                 |
| Anomaly class       | ~1.5%                  |
| Train/Test split    | 80% / 20%              |

### Input Columns

| Column          | Type     | Description                        |
| --------------- | -------- | ---------------------------------- |
| `building_id`   | Integer  | Unique building identifier         |
| `timestamp`     | DateTime | Date and time of the meter reading |
| `meter_reading` | Numeric  | Raw energy consumption value       |

The complete raw dataset is **not included in this repository**.

A smaller `test_df.csv` is included for demonstrating the prediction pipeline.

---

## Feature Engineering

The model uses normalized consumption, temporal features, lag features, rolling statistics, and cyclical time encoding.

The final model uses **15 engineered features**.

| Feature                | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `meter_reading_scaled` | Building-specific MinMax-scaled meter reading       |
| `lag_1`                | Reading from 1 hour earlier                         |
| `lag_2`                | Reading from 2 hours earlier                        |
| `lag_3`                | Reading from 3 hours earlier                        |
| `lag_24`               | Reading from 24 hours earlier                       |
| `lag_48`               | Reading from 48 hours earlier                       |
| `rolling_mean_24`      | Mean of the rolling 24-reading window               |
| `rolling_std_24`       | Standard deviation of the rolling 24-reading window |
| `rolling_max_24`       | Maximum of the rolling 24-reading window            |
| `hour_sin`             | Cyclical encoding of hour                           |
| `hour_cos`             | Cyclical encoding of hour                           |
| `day_sin`              | Cyclical encoding of day                            |
| `day_cos`              | Cyclical encoding of day                            |
| `month_sin`            | Cyclical encoding of month                          |
| `month_cos`            | Cyclical encoding of month                          |

### Temporal Features

Lag features provide historical context for each meter reading:

```text
Current Reading
      │
      ├── lag_1
      ├── lag_2
      ├── lag_3
      ├── lag_24
      └── lag_48
```

Rolling statistics capture local consumption behavior:

```text
Previous 24 readings
        │
        ├── Rolling Mean
        ├── Rolling Standard Deviation
        └── Rolling Maximum
```

Cyclical encoding is used so that time-based features preserve their periodic relationships.

---

## Building-Specific Normalization

Energy consumption can naturally differ significantly between buildings.

Therefore, a separate `MinMaxScaler` is fitted for each building using its normal consumption data.

This allows the model to learn deviations relative to each building's normal consumption pattern rather than relying only on absolute meter-reading values.

The resulting scalers are stored in:

```text
scaler.pkl
```

The trained scaler artifact is hosted on Hugging Face together with the model.

---

## Machine Learning Model

The selected model is a:

**Random Forest Classifier**

Random Forest was selected because it performed well on the engineered tabular features and provided a strong balance between anomaly precision and recall.

### Class Weight

Because the anomaly class is significantly smaller than the normal class, class weighting was used during training:

```python
class_weight={0: 1, 1: 10}
```

This gives greater importance to anomalous observations during model training.

---

## Model Comparison

Several approaches were evaluated for anomaly detection.

| Model                   | Precision |   Recall | F1 — Anomaly | Macro F1 |
| ----------------------- | --------: | -------: | -----------: | -------: |
| Transformer Autoencoder |      0.30 |     0.02 |         0.04 |     0.50 |
| LSTM Autoencoder        |      0.19 |     0.28 |         0.23 |     0.58 |
| XGBoost                 |      0.51 |     0.89 |         0.65 |     0.76 |
| LightGBM                |      0.49 |     0.90 |         0.64 |     0.75 |
| **Random Forest**       |  **0.81** | **0.89** |     **0.85** | **0.92** |

Random Forest achieved the highest anomaly-class F1 score among the evaluated approaches.

---

## Threshold Optimization

Because the dataset is highly imbalanced, the default classification threshold of `0.50` did not provide the desired balance between precision and recall.

A systematic threshold sweep was performed.

| Threshold | Precision |   Recall | F1 Score |
| --------: | --------: | -------: | -------: |
|      0.50 |      0.97 |     0.68 |     0.80 |
|      0.40 |      0.96 |     0.72 |     0.82 |
|      0.30 |      0.94 |     0.76 |     0.84 |
|  **0.20** |  **0.81** | **0.89** | **0.85** |
|      0.15 |      0.84 |     0.83 |     0.83 |

The selected threshold is:

```text
0.20
```

The threshold is stored in:

```text
models/config.json
```

and is used during inference.

---

## Final Model Performance

### Global Classification Report

**Threshold = 0.20**

| Class            | Precision |   Recall | F1 Score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| Normal (0)       |      1.00 |     1.00 |     1.00 |   341,304 |
| **Anomaly (1)**  |  **0.81** | **0.89** | **0.85** | **6,675** |
| Accuracy         |         — |        — | **0.99** |   347,979 |
| Macro Average    |      0.90 |     0.94 | **0.92** |   347,979 |
| Weighted Average |      0.99 |     0.99 | **0.99** |   347,979 |

### Per-Building Performance

| Metric                   |      Value |
| ------------------------ | ---------: |
| Mean F1 across buildings |       0.96 |
| Buildings with F1 = 1.00 |        30+ |
| Buildings with F1 ≥ 0.90 | ~185 / 200 |
| Buildings with F1 < 0.85 |   10 / 200 |
| Minimum F1               |       0.76 |
| Maximum F1               |       1.00 |

### Sample Per-Building Results

| Building ID | F1 Score | Anomaly Count |
| ----------: | -------: | ------------: |
|         941 |     1.00 |            16 |
|         879 |     1.00 |            20 |
|         117 |     1.00 |            44 |
|         136 |     1.00 |            50 |
|          32 |     0.92 |            12 |
|          69 |     0.81 |            39 |
|         240 |     0.76 |            23 |
|        1242 |     0.78 |            20 |

---

## Repository Structure

```text
Energy_Consumption_Anomaly_Detection/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── dataset/
│   └── test_df.csv
│
├── models/
│   ├── config.json
│   ├── rf_model.pkl
│   └── scaler.pkl
│
└── src/
    ├── analysis.ipynb
    ├── calculation.py
    ├── data_preprocess.py
    ├── generate_test.py
    ├── predict.py
    └── training.py
```

---

## Source Code

### `training.py`

Trains the Random Forest classifier using the engineered energy-consumption features.

### `data_preprocess.py`

Contains the preprocessing and feature-engineering pipeline used to transform raw meter data into model-ready features.

### `calculation.py`

Contains supporting calculations used during the processing pipeline.

### `generate_test.py`

Generates a smaller sample dataset for testing the prediction pipeline.

### `predict.py`

Loads the trained model artifacts, processes the sample input data, generates the required features, and performs anomaly prediction.

### `analysis.ipynb`

Contains exploratory data analysis, feature analysis, model evaluation, and experimentation performed during development.

---

## Trained Model

The trained model is hosted on Hugging Face:

**[Energy Consumption Anomaly Detection — Hugging Face](https://huggingface.co/Vikaash17/Energy-Consumption-Anomaly-Detection)**

The model repository contains the trained artifacts required for inference:

```text
rf_model.pkl
scaler.pkl
config.json
```

Keeping the trained artifacts separately from the source repository avoids unnecessarily increasing the size of the GitHub repository while still providing a reproducible inference workflow.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Vikaash-17/Energy_Consumption_Anomaly_Detection.git
cd Energy_Consumption_Anomaly_Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run prediction

```bash
python src/predict.py
```

The prediction pipeline performs the following steps:

```text
Sample CSV
    │
    ▼
Load Data
    │
    ▼
Preprocessing
    │
    ▼
Temporal Features
    │
    ▼
Lag Features
    │
    ▼
Rolling Statistics
    │
    ▼
Building-Specific Scaling
    │
    ▼
Random Forest
    │
    ▼
Anomaly Probability
    │
    ▼
Threshold = 0.20
    │
    ▼
Normal / Anomaly
```

---

## Prediction Input

The prediction pipeline expects the following columns:

```text
building_id
timestamp
meter_reading
```

Example:

```csv
building_id,timestamp,meter_reading
32,2016-01-01 00:00:00,125.42
32,2016-01-01 01:00:00,127.18
32,2016-01-01 02:00:00,129.74
```

The remaining model features are generated automatically by the preprocessing pipeline.

---

## Technologies Used

| Category             | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python 3.x                 |
| Machine Learning     | Scikit-learn               |
| Model                | Random Forest Classifier   |
| Data Processing      | Pandas, NumPy              |
| Model Serialization  | Joblib                     |
| Model Hosting        | Hugging Face Hub           |
| Visualization        | Matplotlib, Seaborn        |
| Development          | Jupyter Notebook / VS Code |

---

## Key Features

* Machine learning-based energy anomaly detection
* Random Forest classification
* Building-specific MinMax normalization
* Lag-based temporal features
* Rolling statistical features
* Cyclical time encoding
* Class imbalance handling
* Precision-recall based threshold optimization
* Bulk CSV prediction
* Hugging Face model hosting
* Reusable preprocessing pipeline
* Per-building performance evaluation
* Lightweight GitHub source repository

---

## Project Outcome

The project demonstrates that engineered temporal and statistical features combined with a tuned Random Forest classifier can effectively identify abnormal energy consumption patterns in highly imbalanced building-level energy data.

The selected model achieved:

```text
Anomaly Precision : 0.81
Anomaly Recall    : 0.89
Anomaly F1 Score  : 0.85
Macro F1          : 0.92
Accuracy          : 0.99
```

using an optimized classification threshold of:

```text
0.20
```

The project also demonstrates a practical ML workflow in which:

**GitHub** contains the source code, preprocessing pipeline, training code, sample data, and documentation, while **Hugging Face** hosts the trained model artifacts used for inference.

---

## Links

### GitHub

**Energy Consumption Anomaly Detection**

https://github.com/Vikaash-17/Energy_Consumption_Anomaly_Detection

### Hugging Face

**Trained Energy Anomaly Detection Model**

https://huggingface.co/Vikaash17/Energy-Consumption-Anomaly-Detection
