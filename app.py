import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Student Performance Prediction", page_icon="🎓", layout="wide")

st.title("🎓 Smart Student Performance Prediction")
st.caption("An interactive academic project demo using Python, Pandas, Scikit-learn, Plotly and Streamlit.")

st.info(
    "Demo transparency: the built-in dataset is illustrative synthetic data created for demonstrating the application. "
    "Its model scores are not research findings and should not be presented as validated results. "
    "Upload your actual project dataset to run an analysis on your data."
)

@st.cache_data
def load_demo_data():
    return pd.read_csv("sample_student_data.csv")

def grade(score):
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload student CSV (optional)", type=["csv"])
    st.markdown("---")
    st.write("Expected example columns:")
    st.code("study_hours\\nattendance_percentage\\nassignment_completion_percentage\\nprevious_score\\nsleep_hours\\nclass_participation\\nfinal_score")

try:
    data = pd.read_csv(uploaded) if uploaded is not None else load_demo_data()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The selected dataset is empty.")
    st.stop()

st.subheader("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Student records", f"{len(data):,}")
c2.metric("Columns", len(data.columns))
c3.metric("Missing values", int(data.isna().sum().sum()))
st.dataframe(data.head(10), use_container_width=True)

numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.warning("Please upload a CSV containing at least two numeric columns.")
    st.stop()

default_target = "final_score" if "final_score" in numeric_cols else numeric_cols[-1]
target = st.selectbox("Select the numeric target column to predict", numeric_cols, index=numeric_cols.index(default_target))
feature_options = [c for c in numeric_cols if c != target]
default_features = [c for c in feature_options if c in [
    "study_hours", "attendance_percentage", "assignment_completion_percentage",
    "previous_score", "sleep_hours", "class_participation"
]]
features = st.multiselect("Select numeric input features", feature_options, default=default_features or feature_options)

if not features:
    st.warning("Select at least one input feature.")
    st.stop()

model_data = data[features + [target]].dropna()
if len(model_data) < 10:
    st.error("At least 10 complete records are required for a basic train/test demonstration.")
    st.stop()

X = model_data[features]
y = model_data[target]
if y.nunique() < 2:
    st.error("The target column must contain more than one distinct value.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

st.subheader("2. Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R² score", f"{r2:.3f}")
st.caption("Metrics are calculated on a held-out test split of the currently selected dataset. They are illustrative when using the built-in synthetic dataset.")

results = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": pred})
fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs. predicted values",
                 labels={"Actual": f"Actual {target}", "Predicted": f"Predicted {target}"})
low = float(min(results.min()))
high = float(max(results.max()))
fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash"))
st.plotly_chart(fig, use_container_width=True)

importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
fig_imp = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature importance (Random Forest)")
st.plotly_chart(fig_imp, use_container_width=True)

st.subheader("3. Explore student performance")
if "final_score" in data.columns:
    fig_hist = px.histogram(data, x="final_score", nbins=20, title="Distribution of final scores")
    st.plotly_chart(fig_hist, use_container_width=True)
if "study_hours" in data.columns and "final_score" in data.columns:
    fig_scatter = px.scatter(data, x="study_hours", y="final_score", trendline="ols",
                             title="Study hours vs. final score")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("4. Predict an individual student's performance")
st.write("Enter values for the selected input features.")
input_values = {}
cols = st.columns(2)
for i, feature in enumerate(features):
    series = model_data[feature]
    min_v, max_v = float(series.min()), float(series.max())
    mean_v = float(series.median())
    with cols[i % 2]:
        input_values[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=min_v,
            max_value=max_v,
            value=min(max(mean_v, min_v), max_v),
            key=f"input_{feature}"
        )

if st.button("Predict student performance", type="primary"):
    one_student = pd.DataFrame([input_values], columns=features)
    prediction = float(model.predict(one_student)[0])
    st.success(f"Predicted {target.replace('_', ' ').title()}: {prediction:.1f}")
    if target == "final_score":
        st.metric("Predicted grade", grade(prediction))
    st.caption("This is a model estimate, not a guarantee or a basis for high-stakes academic decisions.")

st.subheader("5. Project workflow")
st.markdown("""
1. Load the student dataset or upload a CSV.
2. Select a numeric target and relevant input features.
3. Clean incomplete rows for the selected columns.
4. Split records into training and testing sets.
5. Train a Random Forest regression model.
6. Evaluate predictions and inspect feature importance.
7. Enter a student's details to generate an estimated score.
""")
st.caption("Academic prototype. Validate with an appropriate real dataset, document limitations, and obtain permission before using student data.")
