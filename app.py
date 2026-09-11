
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            rgba(49, 51, 63, 0.95),
            rgba(30, 32, 45, 0.95)
        );
        border: 1px solid rgba(255,255,255,0.10);
    }

    .hero h1 {
        margin-bottom: 0.3rem;
        font-size: 2.4rem;
        color: white;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.9;
        color: white;
    }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        opacity: 0.65;
        font-size: 0.85rem;
    }

    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.keras"
COLUMN_TRANSFORMER_PATH = BASE_DIR / "models" / "column_transformer.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


# ============================================================
# LOAD PREPROCESSORS
# ============================================================

@st.cache_resource
def load_preprocessors():

    objects = {}

    if COLUMN_TRANSFORMER_PATH.exists():

        with open(COLUMN_TRANSFORMER_PATH, "rb") as file:
            objects["column_transformer"] = pickle.load(file)

    if SCALER_PATH.exists():

        with open(SCALER_PATH, "rb") as file:
            objects["scaler"] = pickle.load(file)

    return objects


model = load_model()
preprocessors = load_preprocessors()


# ============================================================
# HERO HEADER
# ============================================================

st.html("""
<style>
.hero-card {
    background: linear-gradient(135deg, #202333 0%, #343746 100%);
    padding: 32px 38px;
    border-radius: 20px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    color: white;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 17px;
    color: #d9dbe5;
    line-height: 1.6;
    margin-bottom: 22px;
}

.badge-container {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 22px;
}

.badge {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 13px;
    color: white;
}

.developer {
    font-size: 14px;
    color: #bfc3d1;
    border-top: 1px solid rgba(255,255,255,0.15);
    padding-top: 16px;
}

.developer strong {
    color: white;
}
</style>

<div class="hero-card">

    <div class="hero-title">
        📊 Customer Churn Intelligence
    </div>

    <div class="hero-subtitle">
        Predict customer churn using an
        <strong>Artificial Neural Network (ANN)</strong>
        and identify customers who may need retention attention.
    </div>

    <div class="badge-container">

        <div class="badge">🧠 Artificial Neural Network</div>
        <div class="badge">📈 Churn Prediction</div>
        <div class="badge">🎯 Risk Analysis</div>
        <div class="badge">⚡ Real-Time Prediction</div>

    </div>

    <div class="developer">
        Developed by <strong>Ajim Mujawar</strong>
        &nbsp; • &nbsp;
        Data Science & AI Project
    </div>

</div>
""")


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander("ℹ️ About this project"):

    st.markdown(
        """
        ### Customer Churn Prediction using ANN

        This application uses an **Artificial Neural Network (ANN)**
        built with **TensorFlow/Keras** to estimate whether a bank
        customer is likely to churn.

        **Machine Learning Pipeline**

        `Customer Data → Preprocessing → Scaling → ANN → Probability → Prediction`

        **Technologies**

        - Python
        - TensorFlow / Keras
        - Scikit-learn
        - Pandas
        - NumPy
        - Streamlit

        **Problem Type:** Binary Classification

        **Output:** Churn / No Churn
        """
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👤 Customer Profile")

st.sidebar.markdown(
    "Enter customer information below to generate a churn prediction."
)

st.sidebar.divider()

credit_score = st.sidebar.number_input(
    "💳 Credit Score",
    min_value=300,
    max_value=900,
    value=650,
    step=1
)

geography = st.sidebar.selectbox(
    "🌍 Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "👤 Gender",
    ["Female", "Male"]
)

age = st.sidebar.number_input(
    "🎂 Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

tenure = st.sidebar.number_input(
    "📅 Tenure",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)

balance = st.sidebar.number_input(
    "💰 Account Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
    format="%.2f"
)

num_of_products = st.sidebar.number_input(
    "📦 Number of Products",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)

has_cr_card = st.sidebar.selectbox(
    "💳 Has Credit Card?",
    ["Yes", "No"]
)

is_active_member = st.sidebar.selectbox(
    "⚡ Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.sidebar.number_input(
    "💵 Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
    format="%.2f"
)

st.sidebar.divider()

predict_button = st.sidebar.button(
    "🔮 Predict Customer Churn",
    use_container_width=True,
    type="primary"
)

# ============================================================
# CUSTOMER OVERVIEW
# ============================================================

st.subheader("Customer Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Credit Score", f"{credit_score}")

with col2:
    st.metric("Age", f"{age} years")

with col3:
    st.metric("Balance", f"₹{balance:,.0f}")

with col4:
    st.metric("Products", f"{num_of_products}")

st.divider()

# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input():

    credit_card_value = 1 if has_cr_card == "Yes" else 0
    active_member_value = 1 if is_active_member == "Yes" else 0

    user_data = pd.DataFrame(
        [[
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            num_of_products,
            credit_card_value,
            active_member_value,
            estimated_salary
        ]],
        columns=[
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]
    )

    # --------------------------------------------------------
    # Check preprocessing objects
    # --------------------------------------------------------

    if "column_transformer" not in preprocessors:

        raise FileNotFoundError(
            "column_transformer.pkl was not found in the models folder."
        )

    if "scaler" not in preprocessors:

        raise FileNotFoundError(
            "scaler.pkl was not found in the models folder."
        )

    # --------------------------------------------------------
    # Column Transformer
    # --------------------------------------------------------

    transformed_data = preprocessors[
        "column_transformer"
    ].transform(user_data)

    # Convert sparse matrix if necessary
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    transformed_data = np.asarray(
        transformed_data,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    scaled_data = preprocessors[
        "scaler"
    ].transform(transformed_data)

    scaled_data = np.asarray(
        scaled_data,
        dtype=np.float32
    )

    return scaled_data, user_data


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if model is None:

        st.error(
            """
            ⚠️ Trained model not found.

            Expected:

            `models/churn_model.keras`
            """
        )

    elif (
        "column_transformer" not in preprocessors
        or "scaler" not in preprocessors
    ):

        st.error(
            """
            ⚠️ Preprocessing files are missing.

            Required files:

            `models/column_transformer.pkl`

            `models/scaler.pkl`
            """
        )

    else:

        try:

            # ------------------------------------------------
            # PREPARE INPUT
            # ------------------------------------------------

            user_data, original_input = prepare_input()

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            raw_prediction = model.predict(
                user_data,
                verbose=0
            )

            # Convert model output safely
            probability = float(
                np.asarray(raw_prediction).reshape(-1)[0]
            )

            st.write("DEBUG - ANN Output:", probability)
            st.write("DEBUG - Raw Prediction:", raw_prediction)

            # Make sure probability is between 0 and 1
            probability = float(
                np.clip(probability, 0.0, 1.0)
            )

            churn_probability = probability * 100
            retention_probability = (1 - probability) * 100

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            threshold = 0.50

            if probability >= threshold:

                result = "Likely to Churn"
                emoji = "⚠️"

                recommendation = (
                    "This customer shows a higher probability "
                    "of leaving. Consider proactive retention strategies."
                )

            else:

                result = "Likely to Stay"
                emoji = "✅"

                recommendation = (
                    "This customer currently shows a lower "
                    "probability of churn."
                )

            # =================================================
            # RESULT
            # =================================================

            st.subheader("🔮 Prediction Result")

            if probability >= threshold:

                st.error(
                    f"{emoji} **{result}**"
                )

            else:

                st.success(
                    f"{emoji} **{result}**"
                )

            # =================================================
            # PROBABILITY
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Retention Probability",
                    f"{retention_probability:.2f}%"
                )

            # =================================================
            # RISK LEVEL
            # =================================================

            st.markdown("### 📊 Churn Risk")

            st.progress(
                probability,
                text=f"Churn probability: {churn_probability:.2f}%"
            )

            if probability >= 0.75:

                risk_level = "🔴 High Risk"

            elif probability >= 0.50:

                risk_level = "🟠 Medium Risk"

            elif probability >= 0.25:

                risk_level = "🟡 Low Risk"

            else:

                risk_level = "🟢 Very Low Risk"

            st.info(
                f"**Risk Level:** {risk_level}"
            )

            # =================================================
            # MODEL DIAGNOSTIC
            # =================================================

            with st.expander("🔧 Model Diagnostic"):

                st.write(
                    "**Raw ANN output:**",
                    raw_prediction
                )

                st.write(
                    "**Extracted probability:**",
                    probability
                )

                st.write(
                    "**Classification threshold:**",
                    threshold
                )

                st.write(
                    "**Processed input shape:**",
                    user_data.shape
                )

                st.write(
                    "**Processed input:**"
                )

                st.dataframe(
                    pd.DataFrame(user_data),
                    use_container_width=True
                )

            # =================================================
            # BUSINESS RECOMMENDATION
            # =================================================

            st.markdown("### 💡 Business Recommendation")

            st.write(recommendation)

            if probability >= 0.50:

                st.markdown(
                    """
                    **Suggested retention actions:**

                    - Contact the customer proactively
                    - Review product satisfaction
                    - Offer personalized benefits
                    - Investigate account activity
                    - Consider targeted retention campaigns
                    """
                )

            else:

                st.markdown(
                    """
                    **Suggested actions:**

                    - Maintain customer engagement
                    - Continue personalized communication
                    - Monitor changes in customer activity
                    - Consider loyalty programs
                    """
                )

            # =================================================
            # CUSTOMER DATA
            # =================================================

            with st.expander("🔎 View Customer Input"):

                customer_df = pd.DataFrame(
                    {
                        "Feature": [
                            "Credit Score",
                            "Geography",
                            "Gender",
                            "Age",
                            "Tenure",
                            "Balance",
                            "Number of Products",
                            "Has Credit Card",
                            "Active Member",
                            "Estimated Salary"
                        ],
                        "Value": [
                            credit_score,
                            geography,
                            gender,
                            age,
                            tenure,
                            f"₹{balance:,.2f}",
                            num_of_products,
                            has_cr_card,
                            is_active_member,
                            f"₹{estimated_salary:,.2f}"
                        ]
                    }
                )

                st.dataframe(
                    customer_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.warning(
                """
                This usually means that the preprocessing used
                during deployment does not exactly match the
                preprocessing used during ANN training.
                """
            )

            st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>Customer Churn Intelligence</b><br>
        Developed by <b>Ajim Mujawar</b><br>
        Built with Python • TensorFlow • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
