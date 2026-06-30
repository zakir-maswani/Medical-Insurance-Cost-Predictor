<div align="center">

# 🩺 Medical Insurance Cost Prediction

### Predicting annual medical insurance charges from patient attributes using a Random Forest regression pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

[Overview](#-overview) •
[Dataset](#-dataset) •
[Installation](#-installation) •
[Usage](#-usage) •
[Pipeline](#-data--modeling-pipeline) •
[App](#-streamlit-app) •
[Results](#-model-evaluation)

</div>

---

## 📌 Overview

Health insurers price premiums based on a small set of well-known risk factors: age, body mass index, smoking status, and a handful of demographic details. This project builds an **end-to-end machine learning pipeline** that learns those relationships from historical billing data and predicts the **annual medical insurance charges** for a new patient profile.

The repo contains three things:

| Component | What it does |
|---|---|
| 🧪 **Training notebook/script** | Loads, explores, visualizes, and models the insurance dataset, then exports a trained pipeline |
| 🌐 **Streamlit app** | A lightweight web UI where anyone can enter a patient profile and get an instant cost estimate |
| 📦 **Reusable artifacts** | A serialized `scikit-learn` pipeline + metadata file so the app never has to retrain |

> **Disclaimer:** This project is for educational and portfolio purposes only. Predictions are estimates derived from a public dataset and must **not** be used for actual underwriting, billing, or medical/financial decisions.

---

## 📊 Dataset

The model is trained on the classic **Medical Cost Personal Dataset** (`insurance.csv`), a widely-used benchmark for regression practice originally compiled for *Machine Learning with R* (Brett Lantz) and popularized on Kaggle by Miri Choi.

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age of the primary beneficiary (years) |
| `sex` | categorical | Biological sex (`male` / `female`) |
| `bmi` | numeric | Body Mass Index (kg/m²) |
| `children` | numeric | Number of dependents covered by the plan |
| `smoker` | categorical | Smoking status (`yes` / `no`) |
| `region` | categorical | US residential region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | numeric (**target**) | Individual medical costs billed by health insurance ($) |

1,338 rows, 7 columns, no missing values in the canonical version of the dataset — though the pipeline still includes imputation as a defensive measure for messier real-world data.

---

## 🗂 Project Structure

```
medical-insurance-cost-prediction/
│
├── insurance.csv                       # Raw dataset (not included — see Dataset section)
├── data_preprocessing_and_model_training.ipynb   # EDA, preprocessing, training, evaluation
│
├── app.py                              # Streamlit web app
├── requirements.txt                    # Python dependencies
│
├── delivery_time_model.pkl             # Serialized scikit-learn pipeline (generated after training)
├── model_metadata.json                 # Valid input ranges/options for the app (generated after training)
│
└── README.md                           # You are here
```

> **Note:** The exported model file is named `delivery_time_model.pkl` to match the filename produced by the training script in this repo. Feel free to rename it (and update the corresponding path in `app.py`) to something like `insurance_cost_model.pkl` if you'd like the naming to be project-accurate.

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place `insurance.csv` in the project root. The dataset is available from [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance).

---

## 🚀 Usage

### Train the model

Run the notebook (or the equivalent `.py` script) top to bottom:

```bash
jupyter notebook data_preprocessing_and_model_training.ipynb
```

This will:
1. Load and explore `insurance.csv`
2. Generate visualizations (charges by sex, smoking status, region, BMI, age, etc.)
3. Build a preprocessing + Random Forest pipeline
4. Train, evaluate, and print MSE / MAE / R² metrics
5. Export two artifacts to the project root:
   - `delivery_time_model.pkl` — the fitted pipeline
   - `model_metadata.json` — valid input ranges/categories used to build the app's form

### Launch the app

Once the artifacts above exist:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 🔬 Data & Modeling Pipeline

The model is wrapped in a single `scikit-learn` `Pipeline` so preprocessing and inference always stay in sync.

```
Raw input (age, sex, bmi, children, smoker, region)
        │
        ▼
┌───────────────────────────────────────────┐
│              ColumnTransformer             │
│                                             │
│  Numeric branch          Categorical branch│
│  (age, bmi, children)    (sex, smoker,     │
│  ├─ Median imputation     region)          │
│                          ├─ Most-frequent  │
│                            imputation      │
│                          └─ One-Hot Encode │
└───────────────────────────────────────────┘
        │
        ▼
        RandomForestRegressor (random_state=42)
        │
        ▼
   Predicted annual charges ($)
```

**Why these choices?**
- **Median / most-frequent imputation** — robust to outliers and safe defaults if production data ever contains nulls, even though the source dataset is complete.
- **One-Hot Encoding with `handle_unknown="ignore"`** — prevents the pipeline from breaking if it ever sees a category it wasn't trained on.
- **Random Forest Regressor** — handles non-linear interactions (e.g., the outsized effect of smoking + high BMI together) without manual feature engineering, and is robust to the mild skew in the `charges` target.

**Train/test split:** 70% train / 30% test, `random_state=42` for reproducibility.

---

## 📈 Model Evaluation

The notebook reports three standard regression metrics on both the train and test sets:

| Metric | What it measures |
|---|---|
| **MSE** (Mean Squared Error) | Average squared difference between predicted and actual charges — penalizes large errors heavily |
| **MAE** (Mean Absolute Error) | Average absolute dollar error — easy to interpret directly in dollars |
| **R²** (Coefficient of Determination) | Proportion of variance in charges explained by the model (closer to 1 is better) |

Exact values depend on your training run — they're printed at the end of the notebook. As a rule of thumb for this dataset, a well-tuned Random Forest typically explains **80–88%** of the variance in charges on the held-out test set, with the dominant predictive signal coming from `smoker` status, `age`, and `bmi`.

> 💡 **Improvement idea:** `GridSearchCV` is already imported in the notebook but not yet wired up. Hyperparameter tuning over `n_estimators`, `max_depth`, and `min_samples_leaf` is a natural next step to squeeze out additional performance.

---

## 🌐 Streamlit App

The app (`app.py`) provides a friendly interface around the trained pipeline:

- **Sidebar "Patient Profile" form** — sliders and dropdowns for age, sex, BMI, children, smoking status, and region, auto-populated from `model_metadata.json` so the inputs always stay valid for whatever range the model was trained on.
- **Styled estimate card** — the prediction is rendered as a custom-styled HTML/CSS "insurance estimate card" rather than a plain number, complete with a monthly-equivalent breakdown.
- **Context badges** — automatic BMI category (underweight / normal / overweight / obese) and a smoker-risk indicator, so the estimate is easy to interpret at a glance.
- **Graceful fallbacks** — if the model or metadata files aren't found yet, the app explains exactly what to run first instead of crashing.

---

## 🛠 Tech Stack

- **Language:** Python 3.9+
- **Data handling:** pandas, NumPy
- **Modeling:** scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)
- **Visualization:** Matplotlib, Seaborn
- **Serialization:** joblib, JSON
- **Web app:** Streamlit + custom HTML/CSS

---

## 🗺 Future Improvements

- Wire up `GridSearchCV` for hyperparameter tuning (already imported, unused)
- Add cross-validation instead of a single train/test split for more robust metrics
- Compare Random Forest against Gradient Boosting / XGBoost / linear baselines
- Add SHAP-based feature importance to the app for per-prediction explainability
- Containerize the app with Docker for one-command deployment
- Add automated tests for the preprocessing pipeline

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, new features, or documentation improvements.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- Dataset: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) by Miri Choi on Kaggle, originally from *Machine Learning with R* by Brett Lantz.
- Built with [scikit-learn](https://scikit-learn.org/) and [Streamlit](https://streamlit.io/).

<div align="center">

Made with ☕ and `RandomForestRegressor`

</div>
