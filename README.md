# Road Accident Severity Prediction

A machine learning project that predicts the **severity of a road accident** (Fatal, Serious or Slight injury) from details such as driver profile, vehicle, road, weather and collision conditions.

The whole workflow — cleaning, encoding, feature selection and the model — is wrapped in a single scikit-learn `Pipeline`, so new raw data can be passed in and a prediction comes out directly.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Making Predictions](#making-predictions)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Overview

| Item | Details |
|------|---------|
| **Problem** | Multi-class classification |
| **Target** | `Accident_severity` (Fatal Injury, Serious Injury, Slight Injury) |
| **Model** | Random Forest Classifier |
| **Imbalance handling** | Random oversampling |
| **Feature selection** | Chi-squared (`SelectKBest`, k = 50) |
| **Output** | Trained pipeline saved as `10 pipe.pkl` |

---

## Features

- Exploratory data analysis with bar charts and count plots
- Extracts **Hour of Day** from the accident time
- Balances the severity classes using `RandomOverSampler`
- Fills missing values with `SimpleImputer` (most frequent / constant)
- One-hot encodes categorical columns
- Selects the 50 most relevant features with the chi-squared test
- Trains a Random Forest classifier
- Reports accuracy, classification report and confusion matrix
- Saves the full pipeline with `pickle` for reuse (for example in a website)
- Includes a ready-to-use `predict_severity()` function

---

## Project Structure

```
python practice.py/
│
├── pipeline.py        # Main script: EDA, training, evaluation, prediction
├── pipeline.csv       # Dataset (tab-separated)
├── 10 pipe.pkl        # Trained pipeline (created after running the script)
└── README.md          # Project documentation
```

---

## Dataset

The data is read from `pipeline.csv` (tab-separated) and contains road accident records with the following kinds of information:

| Category | Example columns |
|----------|-----------------|
| **Time** | `Time`, `Day_of_week` |
| **Driver** | `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Driving_experience`, `Vehicle_driver_relation` |
| **Vehicle** | `Type_of_vehicle`, `Owner_of_vehicle`, `Service_year_of_vehicle`, `Defect_of_vehicle`, `Vehicle_movement` |
| **Road** | `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`, `Road_surface_type`, `Road_surface_conditions` |
| **Environment** | `Light_conditions`, `Weather_conditions` |
| **Accident** | `Type_of_collision`, `Number_of_vehicles_involved`, `Number_of_casualties`, `Cause_of_accident` |
| **Casualty** | `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Casualty_severity`, `Work_of_casuality`, `Fitness_of_casuality`, `Pedestrian_movement` |
| **Target** | `Accident_severity` |

---

## How It Works

```
 Raw data
    │
    ▼
 Extract Hour_of_Day from Time
    │
    ▼
 Encode target (LabelEncoder)
    │
    ▼
 Balance classes (RandomOverSampler)
    │
    ▼
 Train / Test split (80% / 20%)
    │
    ▼
 ┌────────────────── Pipeline ──────────────────┐
 │ 1. Impute missing values (SimpleImputer)     │
 │ 2. One-hot encode columns (OneHotEncoder)    │
 │ 3. Select best 50 features (chi2)            │
 │ 4. Random Forest Classifier                  │
 └──────────────────────────────────────────────┘
    │
    ▼
 Evaluate → Save with pickle → Predict new input
```

### Missing value strategy

| Strategy | Columns |
|----------|---------|
| `most_frequent` | Educational level, Vehicle–driver relation, Driving experience, Type of vehicle, Area, Lanes/Medians, Road alignment, Junction type, Road surface type, Type of collision, Vehicle movement, Work of casualty, Fitness of casualty |
| `constant` (`"Unknown"`) | Service year of vehicle, Defect of vehicle |

---

## Installation

**1. Clone or download the project**

```bash
git clone <your-repo-url>
cd "python practice.py"
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**3. Install the dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Usage

Make sure `pipeline.csv` is in the same folder as `pipeline.py`, then run:

```bash
python pipeline.py
```

The script will:

1. Load the dataset and print its shape and info
2. Show the target distribution and EDA plots (close each plot window to continue)
3. Prepare the data, balance the classes and split into train/test
4. Train the pipeline
5. Print the **accuracy**, **classification report** and **confusion matrix**
6. Save the trained pipeline as `10 pipe.pkl`
7. Run an example prediction

---

## Making Predictions

Use the `predict_severity()` function with the 31 input features:

```python
predicted_class = predict_severity(
    Day_of_week="Thursday",
    Age_band_of_driver="31-50",
    Sex_of_driver="Male",
    Educational_level="Junior high school",
    Vehicle_driver_relation="Owner",
    Driving_experience=None,
    Type_of_vehicle="Long lorry",
    Owner_of_vehicle="Owner",
    Service_year_of_vehicle="Unknown",
    Defect_of_vehicle=None,
    Area_accident_occured="Other",
    Lanes_or_Medians="Two-way (divided with solid lines road marking)",
    Road_allignment="Tangent road with flat terrain",
    Types_of_Junction=None,
    Road_surface_type=None,
    Road_surface_conditions="Dry",
    Light_conditions="Daylight",
    Weather_conditions="Normal",
    Type_of_collision="Collision with animals",
    Number_of_vehicles_involved=2,
    Number_of_casualties=1,
    Vehicle_movement="Going straight",
    Casualty_class="Driver or rider",
    Sex_of_casualty="Male",
    Age_band_of_casualty="18-30",
    Casualty_severity=3,
    Work_of_casuality="Driver",
    Fitness_of_casuality="Normal",
    Pedestrian_movement="Not a Pedestrian",
    Cause_of_accident="Changing lane to the left",
    Hour_of_Day=12,
)

severity_map = {0: "Fatal Injury", 1: "Serious Injury", 2: "Slight Injury"}
print(severity_map[predicted_class[0]])
```

### Severity labels

| Code | Meaning |
|------|---------|
| `0` | Fatal Injury |
| `1` | Serious Injury |
| `2` | Slight Injury |

---

## Tech Stack

| Purpose | Library |
|---------|---------|
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` |
| Class balancing | `imbalanced-learn` |
| Model saving | `pickle` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: pipeline.csv` | Put `pipeline.csv` in the same folder as `pipeline.py`, and run the script from that folder |
| Dataset loads as one big column | The file separator is different. Change `sep="\t"` to `sep=","` in `pd.read_csv` |
| `TypeError: ... unexpected keyword argument 'sparse'` | Newer scikit-learn versions renamed it. Use `OneHotEncoder(sparse_output=False, ...)` instead of `sparse=False` |
| `ModuleNotFoundError: imblearn` | Run `pip install imbalanced-learn` |
| Plots block the script | Close each plot window to let the script continue |

---

## Future Improvements

- Compare more models (XGBoost, Gradient Boosting, Logistic Regression)
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Try SMOTE-style oversampling and class weights
- Build a web app (Flask / Streamlit) that uses `10 pipe.pkl`
- Add feature importance plots

---

## License

This project is for learning and practice purposes. Add a license of your choice (for example MIT) if you plan to share it publicly.
