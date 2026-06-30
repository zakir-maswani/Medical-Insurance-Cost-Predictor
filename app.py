"""
Medical Insurance Cost Predictor
---------------------------------
A Streamlit front-end for the Random Forest pipeline trained in
`data_preprocessing_and_model_training.ipynb`.

Run with:
    streamlit run app.py
"""

import json
import os
import hashlib
from datetime import date

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MediCost | Insurance Estimator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "delivery_time_model.pkl"
METADATA_PATH = "model_metadata.json"

DEFAULT_NUMERIC_RANGES = {"age": [18.0, 64.0], "bmi": [15.0, 55.0], "children": [0.0, 5.0]}
DEFAULT_CATEGORIES = {
    "sex": ["female", "male"],
    "smoker": ["no", "yes"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}

# --------------------------------------------------------------------------
# Styling — fonts, palette, and the signature "estimate card" component
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #123B42 0%, #0B2027 55%, #08171C 100%);
        color: #EAF4F4;
    }

    section[data-testid="stSidebar"] {
        background: #0E2E34;
        border-right: 1px solid rgba(95, 211, 176, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: #CFE6E6 !important;
    }

    /* Hero banner */
    .hero {
        padding: 2.1rem 2.4rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #123B42 0%, #0E2E34 60%, #0B2027 100%);
        border: 1px solid rgba(95, 211, 176, 0.18);
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: #F4FBFA;
        letter-spacing: -0.01em;
    }
    .hero p {
        font-size: 1rem;
        color: #9FB8BA;
        margin: 0;
        max-width: 640px;
    }
    .hero .eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        color: #5FD3B0;
        margin-bottom: 0.6rem;
        display: inline-block;
    }

    /* Section labels in the sidebar */
    .side-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5FD3B0;
        margin: 1.1rem 0 0.3rem 0;
    }

    /* Info / mini summary cards */
    .mini-card {
        background: rgba(95, 211, 176, 0.06);
        border: 1px solid rgba(95, 211, 176, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .mini-card .label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9FB8BA;
        margin-bottom: 0.15rem;
    }
    .mini-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #F4FBFA;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-top: 0.25rem;
    }

    /* Signature element: insurance estimate card with a perforated divider */
    .estimate-card {
        position: relative;
        background: linear-gradient(135deg, #163B40 0%, #0F2C31 100%);
        border: 1px solid rgba(242, 165, 65, 0.35);
        border-radius: 20px;
        padding: 1.8rem 2rem 1.5rem 2rem;
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }
    .estimate-card .ec-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .estimate-card .ec-eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.7rem;
        color: #F2A541;
    }
    .estimate-card .ec-id {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        color: #9FB8BA;
        letter-spacing: 0.04em;
    }
    .estimate-card .ec-amount {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #F4FBFA;
        margin: 0.6rem 0 0.1rem 0;
        letter-spacing: -0.02em;
    }
    .estimate-card .ec-amount-label {
        font-size: 0.85rem;
        color: #9FB8BA;
        margin-bottom: 1.1rem;
    }
    .ec-divider {
        border: none;
        border-top: 2px dashed rgba(159, 184, 186, 0.3);
        margin: 1.1rem 0;
        position: relative;
    }
    .ec-divider::before, .ec-divider::after {
        content: "";
        position: absolute;
        top: -10px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #08171C;
    }
    .ec-divider::before { left: -2rem; }
    .ec-divider::after { right: -2rem; }
    .estimate-card .ec-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }
    .estimate-card .ec-grid .item .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #9FB8BA;
        margin-bottom: 0.15rem;
    }
    .estimate-card .ec-grid .item .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F4FBFA;
    }

    .footnote {
        font-size: 0.78rem;
        color: #6E8A8C;
        margin-top: 1.4rem;
        line-height: 1.5;
    }

    .stButton > button {
        background: #F2A541;
        color: #0B2027;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.2rem;
        font-family: 'Space Grotesk', sans-serif;
        width: 100%;
    }
    .stButton > button:hover {
        background: #F4B968;
        color: #0B2027;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()

numeric_ranges = (metadata or {}).get("numeric_ranges", DEFAULT_NUMERIC_RANGES)
categorical_options = (metadata or {}).get("categorical_options", DEFAULT_CATEGORIES)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">Random Forest · scikit-learn pipeline</span>
        <h1>🩺 Medical Insurance Cost Estimator</h1>
        <p>Enter a patient profile in the sidebar to get an instant, model-based estimate
        of annual medical insurance charges, based on the well-known Medical Cost Personal
        Dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None or metadata is None:
    st.warning(
        "⚠️ Model artifacts not found. Run `data_preprocessing_and_model_training.ipynb` "
        "first — it will generate `delivery_time_model.pkl` and `model_metadata.json` "
        "in this folder. The form below is shown with default ranges in the meantime."
    )

# --------------------------------------------------------------------------
# Sidebar — Patient Profile form
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧾 Patient Profile")
    st.caption("Adjust the fields below, then estimate the premium.")

    st.markdown('<div class="side-label">Demographics</div>', unsafe_allow_html=True)
    age = st.slider(
        "Age",
        min_value=int(numeric_ranges["age"][0]),
        max_value=int(numeric_ranges["age"][1]),
        value=int((numeric_ranges["age"][0] + numeric_ranges["age"][1]) // 2),
    )
    sex = st.selectbox("Sex", options=categorical_options["sex"])
    region = st.selectbox("Region", options=categorical_options["region"])

    st.markdown('<div class="side-label">Health</div>', unsafe_allow_html=True)
    bmi = st.slider(
        "BMI (Body Mass Index)",
        min_value=float(numeric_ranges["bmi"][0]),
        max_value=float(numeric_ranges["bmi"][1]),
        value=round((numeric_ranges["bmi"][0] + numeric_ranges["bmi"][1]) / 2, 1),
        step=0.1,
    )
    smoker = st.selectbox("Smoker", options=categorical_options["smoker"])

    st.markdown('<div class="side-label">Coverage</div>', unsafe_allow_html=True)
    children = st.slider(
        "Number of children / dependents",
        min_value=int(numeric_ranges["children"][0]),
        max_value=int(numeric_ranges["children"][1]),
        value=int(numeric_ranges["children"][0]),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("💲 Estimate Premium", disabled=model is None)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def bmi_category(value: float):
    if value < 18.5:
        return "Underweight", "#5FD3B0", "rgba(95, 211, 176, 0.15)"
    if value < 25:
        return "Normal", "#5FD3B0", "rgba(95, 211, 176, 0.15)"
    if value < 30:
        return "Overweight", "#F2A541", "rgba(242, 165, 65, 0.15)"
    return "Obese", "#EF6F6C", "rgba(239, 111, 108, 0.15)"


def smoker_risk(value: str):
    if value == "yes":
        return "Elevated Risk", "#EF6F6C", "rgba(239, 111, 108, 0.15)"
    return "Standard Risk", "#5FD3B0", "rgba(95, 211, 176, 0.15)"


def estimate_id(profile: dict) -> str:
    raw = json.dumps(profile, sort_keys=True) + str(date.today())
    digest = hashlib.sha256(raw.encode()).hexdigest()[:6].upper()
    return f"EST-{date.today().strftime('%y%m%d')}-{digest}"


# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown("#### Profile Summary")

    bmi_label, bmi_color, bmi_bg = bmi_category(bmi)
    smoke_label, smoke_color, smoke_bg = smoker_risk(smoker)

    st.markdown(
        f"""
        <div class="mini-card">
            <div class="label">Age</div>
            <div class="value">{age} years</div>
        </div>
        <div class="mini-card">
            <div class="label">Sex · Region</div>
            <div class="value">{sex.title()} · {region.title()}</div>
        </div>
        <div class="mini-card">
            <div class="label">Body Mass Index</div>
            <div class="value">{bmi:.1f}
                <span class="badge-pill" style="color:{bmi_color}; background:{bmi_bg};">{bmi_label}</span>
            </div>
        </div>
        <div class="mini-card">
            <div class="label">Smoking Status</div>
            <div class="value">{smoker.title()}
                <span class="badge-pill" style="color:{smoke_color}; background:{smoke_bg};">{smoke_label}</span>
            </div>
        </div>
        <div class="mini-card">
            <div class="label">Dependents</div>
            <div class="value">{children}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("#### Estimate")

    if model is not None and (predict_clicked or "last_prediction" in st.session_state):
        if predict_clicked:
            input_df = pd.DataFrame(
                [{
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                }]
            )
            prediction = float(model.predict(input_df)[0])
            st.session_state["last_prediction"] = prediction
            st.session_state["last_profile"] = input_df.iloc[0].to_dict()

        prediction = st.session_state["last_prediction"]
        profile = st.session_state["last_profile"]
        monthly = prediction / 12
        eid = estimate_id(profile)

        st.markdown(
            f"""
            <div class="estimate-card">
                <div class="ec-top">
                    <span class="ec-eyebrow">Annual Estimate</span>
                    <span class="ec-id">{eid}</span>
                </div>
                <div class="ec-amount">${prediction:,.2f}</div>
                <div class="ec-amount-label">predicted annual medical insurance charges</div>
                <hr class="ec-divider" />
                <div class="ec-grid">
                    <div class="item">
                        <div class="label">Monthly Equivalent</div>
                        <div class="value">${monthly:,.2f}</div>
                    </div>
                    <div class="item">
                        <div class="label">Risk Profile</div>
                        <div class="value">{smoke_label}</div>
                    </div>
                    <div class="item">
                        <div class="label">BMI Category</div>
                        <div class="value">{bmi_label}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="footnote">Estimate generated by a Random Forest Regressor '
            'trained on historical insurance charges. This is a portfolio/educational '
            'tool, not a real quote — actual premiums depend on factors and underwriting '
            'rules not captured here.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("👈 Fill in the patient profile and click **Estimate Premium** to see a prediction.")

# --------------------------------------------------------------------------
# About / model info
# --------------------------------------------------------------------------
with st.expander("ℹ️ About this model"):
    st.markdown(
        """
- **Algorithm:** `RandomForestRegressor` (scikit-learn), wrapped in a `Pipeline` with a
  `ColumnTransformer` for preprocessing.
- **Numeric features** (`age`, `bmi`, `children`): median imputation.
- **Categorical features** (`sex`, `smoker`, `region`): most-frequent imputation +
  one-hot encoding.
- **Training data:** the *Medical Cost Personal Dataset* — 1,338 patient records with
  known billed charges.
- **Strongest predictors:** smoking status, age, and BMI typically drive the largest
  swings in predicted charges.
        """
    )
