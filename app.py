import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "churn_model.pkl"
TRAIN_PATH = "churn-bigml-80.csv"
TEST_PATH = "churn-bigml-20.csv"  # optional: the official held-out test file
RANDOM_STATE = 42


# ==========================================
# HELPERS
# ==========================================

def show_chart(fig):
    """Display a Plotly chart at full width (works on old and new Streamlit)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(data, **kwargs):
    """Display a dataframe at full width (works on old and new Streamlit)."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["Churn"] = (
        data["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .astype(int)
    )
    return data


def prepare_features(data):
    X = data.drop(columns="Churn").copy()
    X["Area code"] = X["Area code"].astype(str)  # match training format
    return X, data["Churn"]


@st.cache_data
def get_test_set(data):
    """Use the official 20% file if present, otherwise split the 80% file."""
    if os.path.exists(TEST_PATH):
        X_test, y_test = prepare_features(load_data(TEST_PATH))
        return X_test, y_test, f"held-out file `{TEST_PATH}`"

    X, y = prepare_features(data)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test, "a 20% split of the training file"


@st.cache_data
def evaluate(_model, X_test, y_test, threshold):
    y_prob = _model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return metrics, cm


@st.cache_data
def get_feature_importance(_model):
    """Return a DataFrame of feature importances, or None if unavailable."""
    try:
        preprocessor = _model.named_steps["preprocessor"]
        classifier = _model.named_steps["classifier"]
        names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
        importances = classifier.feature_importances_
        if len(names) != len(importances):
            return None
        return (
            pd.DataFrame({"Feature": names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return None


def churn_rate_for(data, column, value):
    subset = data[data[column] == value]
    return subset["Churn"].mean() * 100 if len(subset) else float("nan")


def charge_rate(data, minutes_col, charge_col):
    """Typical price per minute, estimated from the data."""
    valid = data[data[minutes_col] > 0]
    return (valid[charge_col] / valid[minutes_col]).median()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Telecom Churn Analytics")

PAGES = [
    "🏠 Dashboard",
    "📈 Churn Analysis",
    "🤖 Predict Churn",
    "📊 Model Performance",
    "🔍 Feature Importance",
    "💡 Business Insights",
]

page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.divider()

threshold = st.sidebar.slider(
    "⚙️ Churn decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help=(
        "A customer is predicted to churn when their churn probability is at "
        "or above this value. Lower it to catch more churners (higher recall, "
        "more false alarms). Used on the Predict Churn and Model Performance pages."
    ),
)

st.sidebar.caption(
    f"Medium risk starts at {threshold * 0.6:.0%}; high risk at {threshold:.0%}."
)


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model from `{MODEL_PATH}`.")
    st.exception(exc)
    st.stop()

try:
    df = load_data(TRAIN_PATH)
except Exception as exc:
    st.error(f"Could not load the dataset from `{TRAIN_PATH}`.")
    st.exception(exc)
    st.stop()

feature_columns = [c for c in df.columns if c != "Churn"]

st.sidebar.divider()
st.sidebar.caption(
    f"Dataset: {len(df):,} customers · Churn rate: {df['Churn'].mean():.1%}"
)


# ==========================================
# PAGE: DASHBOARD
# ==========================================

def render_dashboard():
    total_customers = len(df)
    churned_customers = int(df["Churn"].sum())
    retained_customers = total_customers - churned_customers
    churn_rate = churned_customers / total_customers * 100

    st.title("🏠 Dashboard")
    st.write(
        "A high-level view of customer churn. Use the sidebar to explore the "
        "patterns, predict churn for a customer, or review the model."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("Churned Customers", f"{churned_customers:,}")
    k3.metric("Retained Customers", f"{retained_customers:,}")
    k4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.info(
        f"Out of {total_customers:,} customers, {churned_customers:,} have churned. "
        f"This represents a churn rate of {churn_rate:.2f}%."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")
        distribution = pd.DataFrame(
            {
                "Status": ["Retained", "Churned"],
                "Customers": [retained_customers, churned_customers],
            }
        )
        show_chart(
            px.pie(
                distribution,
                values="Customers",
                names="Status",
                hole=0.4,
            )
        )

    with right:
        st.subheader("Churn by State (Top 10)")
        by_state = df.groupby("State")["Churn"].agg(["mean", "count"])
        by_state = by_state[by_state["count"] >= 10]
        if by_state.empty:
            st.info("Not enough data per state to show this chart.")
        else:
            top_states = (
                by_state["mean"].mul(100).sort_values(ascending=False).head(10).reset_index()
            )
            top_states.columns = ["State", "Churn Rate (%)"]
            show_chart(px.bar(top_states, x="State", y="Churn Rate (%)"))
            st.caption("Only states with at least 10 customers are included.")


# ==========================================
# PAGE: CHURN ANALYSIS
# ==========================================

def render_churn_analysis():
    st.title("📈 Churn Analysis")
    st.write("How churn varies with plans, customer service contact and usage.")

    plan_col1, plan_col2 = st.columns(2)

    with plan_col1:
        st.subheader("Churn by International Plan")
        intl = df.groupby("International plan")["Churn"].mean().mul(100).reset_index()
        intl.columns = ["International Plan", "Churn Rate (%)"]
        show_chart(px.bar(intl, x="International Plan", y="Churn Rate (%)"))

    with plan_col2:
        st.subheader("Churn by Voice Mail Plan")
        vmail = df.groupby("Voice mail plan")["Churn"].mean().mul(100).reset_index()
        vmail.columns = ["Voice Mail Plan", "Churn Rate (%)"]
        show_chart(px.bar(vmail, x="Voice Mail Plan", y="Churn Rate (%)"))

    st.subheader("Churn by Customer Service Calls")
    service = df.groupby("Customer service calls")["Churn"].mean().mul(100).reset_index()
    service.columns = ["Customer Service Calls", "Churn Rate (%)"]
    show_chart(px.bar(service, x="Customer Service Calls", y="Churn Rate (%)"))

    st.subheader("Average Customer Usage")
    usage_columns = [
        "Total day minutes",
        "Total eve minutes",
        "Total night minutes",
        "Total intl minutes",
    ]
    usage = (
        df.groupby("Churn")[usage_columns]
        .mean()
        .rename(index={0: "Retained", 1: "Churned"})
        .reset_index()
        .melt(id_vars="Churn", var_name="Usage Type", value_name="Average Minutes")
    )
    show_chart(
        px.bar(
            usage,
            x="Usage Type",
            y="Average Minutes",
            color="Churn",
            barmode="group",
        )
    )


# ==========================================
# PAGE: PREDICT CHURN
# ==========================================

def render_prediction(threshold):
    st.title("🤖 Predict Churn")
    st.write("Enter a customer's details to estimate their churn probability.")

    medium_threshold = threshold * 0.6

    states = sorted(df["State"].unique())
    area_codes = sorted(df["Area code"].unique())
    default_area_index = area_codes.index(415) if 415 in area_codes else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox("State", states)
        area_code = st.selectbox("Area code", area_codes, index=default_area_index)
        account_length = st.number_input(
            "Account length (days)", min_value=1, max_value=300, value=100, step=1
        )
        international_plan = st.selectbox("International plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
        number_vmail_messages = st.number_input(
            "Number of voicemail messages",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            disabled=(voice_mail_plan == "No"),
            help="Only available with a voice mail plan.",
        )
        if voice_mail_plan == "No":
            number_vmail_messages = 0

    with col2:
        total_day_minutes = st.number_input(
            "Total day minutes", min_value=0.0, max_value=500.0, value=180.0, step=1.0
        )
        total_day_calls = st.number_input(
            "Total day calls", min_value=0, max_value=200, value=100, step=1
        )
        total_eve_minutes = st.number_input(
            "Total evening minutes", min_value=0.0, max_value=500.0, value=200.0, step=1.0
        )
        total_eve_calls = st.number_input(
            "Total evening calls", min_value=0, max_value=200, value=100, step=1
        )

    with col3:
        total_night_minutes = st.number_input(
            "Total night minutes", min_value=0.0, max_value=500.0, value=200.0, step=1.0
        )
        total_night_calls = st.number_input(
            "Total night calls", min_value=0, max_value=200, value=100, step=1
        )
        total_intl_minutes = st.number_input(
            "Total international minutes", min_value=0.0, max_value=100.0, value=10.0, step=0.1
        )
        total_intl_calls = st.number_input(
            "Total international calls", min_value=0, max_value=50, value=4, step=1
        )
        customer_service_calls = st.number_input(
            "Customer service calls", min_value=0, max_value=20, value=1, step=1
        )

    st.caption(
        "Charges are calculated automatically from minutes, using the typical "
        "per-minute rates found in the training data."
    )

    if not st.button("🔮 Predict Churn", type="primary"):
        return

    try:
        day_rate = charge_rate(df, "Total day minutes", "Total day charge")
        eve_rate = charge_rate(df, "Total eve minutes", "Total eve charge")
        night_rate = charge_rate(df, "Total night minutes", "Total night charge")
        intl_rate = charge_rate(df, "Total intl minutes", "Total intl charge")

        customer_data = pd.DataFrame(
            [
                {
                    "State": state,
                    "Account length": account_length,
                    "Area code": str(area_code),
                    "International plan": international_plan,
                    "Voice mail plan": voice_mail_plan,
                    "Number vmail messages": number_vmail_messages,
                    "Total day minutes": total_day_minutes,
                    "Total day calls": total_day_calls,
                    "Total day charge": round(total_day_minutes * day_rate, 2),
                    "Total eve minutes": total_eve_minutes,
                    "Total eve calls": total_eve_calls,
                    "Total eve charge": round(total_eve_minutes * eve_rate, 2),
                    "Total night minutes": total_night_minutes,
                    "Total night calls": total_night_calls,
                    "Total night charge": round(total_night_minutes * night_rate, 2),
                    "Total intl minutes": total_intl_minutes,
                    "Total intl calls": total_intl_calls,
                    "Total intl charge": round(total_intl_minutes * intl_rate, 2),
                    "Customer service calls": customer_service_calls,
                }
            ]
        )[feature_columns]  # same column order as the training data

        st.subheader("Customer Information")
        show_table(customer_data, hide_index=True)

        probability = float(model.predict_proba(customer_data)[0][1])

        st.subheader("Prediction Result")

        if probability >= threshold:
            st.error("⚠️ This customer is predicted to CHURN.")
        else:
            st.success("✅ This customer is predicted NOT to churn.")

        st.metric("Churn Probability", f"{probability:.2%}")

        if probability >= threshold:
            st.error("🔴 High Churn Risk")
        elif probability >= medium_threshold:
            st.warning("🟠 Medium Churn Risk")
        else:
            st.success("🟢 Low Churn Risk")

    except Exception as exc:
        st.error("The model could not make the prediction.")
        st.exception(exc)


# ==========================================
# PAGE: MODEL PERFORMANCE
# ==========================================

def render_performance(threshold):
    st.title("📊 Model Performance")

    try:
        X_test, y_test, test_source = get_test_set(df)
        metrics, cm = evaluate(model, X_test, y_test, threshold)
    except Exception as exc:
        st.warning("Model performance could not be calculated.")
        st.exception(exc)
        return

    st.write(
        "These metrics evaluate the trained model on "
        f"{test_source} ({len(X_test):,} customers), "
        f"using a decision threshold of {threshold:.0%}. "
        "Change the threshold in the sidebar to see how the metrics move."
    )

    for column, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(name, f"{value:.2%}")

    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Customers"},
        x=["Retained", "Churned"],
        y=["Retained", "Churned"],
    )
    show_chart(fig_cm)


# ==========================================
# PAGE: FEATURE IMPORTANCE
# ==========================================

def render_feature_importance():
    st.title("🔍 Feature Importance")

    feature_importance = get_feature_importance(model)

    if feature_importance is None:
        st.info("Feature importance is not available for this model.")
        return

    st.write("These features contributed most to the model's predictions.")
    top_features = feature_importance.head(15)

    show_chart(
        px.bar(
            top_features.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Features Influencing Churn Predictions",
        )
    )

    st.subheader("Top Predictive Features")
    table = top_features.copy()
    table["Importance"] = (table["Importance"] * 100).round(2)
    table = table.rename(columns={"Importance": "Importance (%)"})
    show_table(table, hide_index=True)


# ==========================================
# PAGE: BUSINESS INSIGHTS
# ==========================================

def segment_row(name, mask):
    subset = df[mask]
    rate = subset["Churn"].mean() * 100 if len(subset) else float("nan")
    return {"Segment": name, "Customers": len(subset), "Churn Rate (%)": round(rate, 1)}


def render_business_insights():
    st.title("💡 Business Insights")

    avg_calls = df.groupby("Churn")["Customer service calls"].mean()

    st.subheader("Key Findings")
    st.markdown(
        f"""
- **International plan:** churn is {churn_rate_for(df, "International plan", "Yes"):.1f}% for
  customers with an international plan versus {churn_rate_for(df, "International plan", "No"):.1f}% without one.
- **Customer service:** churned customers make {avg_calls.get(1, float("nan")):.2f} service calls on average,
  compared with {avg_calls.get(0, float("nan")):.2f} for retained customers.
- **Voice mail plan:** churn is {churn_rate_for(df, "Voice mail plan", "Yes"):.1f}% with a voice mail plan
  versus {churn_rate_for(df, "Voice mail plan", "No"):.1f}% without one.
- **Usage:** churned customers show different average calling patterns from retained customers
  (see the Churn Analysis page).
"""
    )

    st.subheader("High-Risk Segments")
    segments = pd.DataFrame(
        [
            segment_row("All customers", df["Churn"].notna()),
            segment_row("International plan", df["International plan"] == "Yes"),
            segment_row("4+ customer service calls", df["Customer service calls"] >= 4),
            segment_row(
                "International plan AND 4+ service calls",
                (df["International plan"] == "Yes") & (df["Customer service calls"] >= 4),
            ),
            segment_row("No voice mail plan", df["Voice mail plan"] == "No"),
        ]
    )
    show_table(segments, hide_index=True)
    st.caption("Segments with few customers can show volatile churn rates.")

    st.subheader("Recommended Actions")
    st.markdown(
        """
- **Prioritise retention outreach** for customers in the high-risk segments above.
- **Review international plan pricing and quality**, since that group churns more.
- **Treat repeated service calls as an early warning** and escalate those customers
  to a retention team before they leave.
- **Use the Predict Churn page** to score individual customers, and tune the
  decision threshold in the sidebar to balance missed churners against false alarms.
"""
    )


# ==========================================
# ROUTING
# ==========================================

if page == PAGES[0]:
    render_dashboard()
elif page == PAGES[1]:
    render_churn_analysis()
elif page == PAGES[2]:
    render_prediction(threshold)
elif page == PAGES[3]:
    render_performance(threshold)
elif page == PAGES[4]:
    render_feature_importance()
else:
    render_business_insights()
