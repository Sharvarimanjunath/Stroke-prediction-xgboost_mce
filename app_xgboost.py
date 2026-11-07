import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stroke_prediction_model.pkl"
)

COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

TEST_COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "test_model_comparison.csv"
)

FINAL_METRICS_PATH = os.path.join(
    MODEL_DIR,
    "final_metrics.csv"
)

THRESHOLD_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "threshold_results.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)

ROC_PATH = os.path.join(
    MODEL_DIR,
    "roc_curve_comparison.png"
)

PR_PATH = os.path.join(
    MODEL_DIR,
    "precision_recall_curve.png"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "threshold_tuning.png"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.png"
)

FEATURE_CSV_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    saved_model = joblib.load(
        MODEL_PATH
    )

except FileNotFoundError:

    st.error(
        "❌ Model file not found.\n\n"
        "Please make sure that "
        "'stroke_prediction_model.pkl' "
        "is inside the models folder."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

if isinstance(saved_model, dict):

    model = saved_model.get(
        "model"
    )

    preprocessor = saved_model.get(
        "preprocessor"
    )

    threshold = saved_model.get(
        "threshold",
        0.50
    )

    model_name = saved_model.get(
        "model_name",
        "XGBoost"
    )

    feature_columns = saved_model.get(
        "features"
    )

else:

    model = saved_model

    preprocessor = None

    threshold = 0.50

    model_name = "XGBoost"

    feature_columns = None


if model is None:

    st.error(
        "❌ Valid machine learning model "
        "was not found inside the pickle file."
    )

    st.stop()


# ============================================================
# LOAD FINAL MODEL METRICS
# ============================================================

final_metrics = None

if os.path.exists(FINAL_METRICS_PATH):

    try:

        metrics_df = pd.read_csv(
            FINAL_METRICS_PATH
        )

        if not metrics_df.empty:

            final_metrics = metrics_df.iloc[0]

    except Exception as e:

        st.warning(
            f"⚠️ Could not load final metrics: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 🧠 Stroke Prediction

        **Purpose:** Predict potential stroke
        risk based on patient health and
        demographic information.
        """
    )

    st.markdown("---")

    st.subheader(
        "🎯 Decision Rule"
    )

    st.write(
        "The optimized decision threshold is:"
    )

    st.metric(
        "Threshold",
        f"{threshold:.2f}"
    )

    st.write(
        f"""
        If the predicted probability is greater
        than or equal to **{threshold:.2f}**, the
        system classifies the patient as having
        predicted stroke risk.
        """
    )

    st.markdown("---")

    st.subheader(
        "🤖 Final Model"
    )

    st.write(
        "XGBoost"
    )

    st.markdown("---")

    st.caption(
        "For academic and research purposes only."
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🧠 Stroke Prediction System"
)

st.subheader(
    "Machine Learning Based Stroke Risk Prediction"
)

st.write(
    """
    Enter the patient's information below to
    estimate the predicted stroke risk using
    the trained XGBoost model.
    """
)

st.info(
    "⚠️ This application is a machine-learning "
    "research prototype and is not a medical diagnosis."
)


# ============================================================
# PATIENT INPUT
# ============================================================

st.header(
    "👤 Patient Information"
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=50.0,
        step=1.0
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    ever_married = st.selectbox(
        "Ever Married",
        [
            "Yes",
            "No"
        ]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=0.1
    )

    bmi = st.number_input(
        "BMI",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

    residence_type = st.selectbox(
        "Residence Type",
        [
            "Urban",
            "Rural"
        ]
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown"
        ]
    )
# ============================================================
# INITIALIZE PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Stroke Risk",
    type="primary",
    width="stretch"
)


# ============================================================
# PREDICTION
# ============================================================

# ============================================================
# INPUT VALIDATION
# ============================================================

if predict_button and (
    age <= 0
    or avg_glucose_level <= 0
    or bmi <= 0
):
    st.error(
        "❌ Invalid Patient Information. "
        "Please enter valid positive values for "
        "Age, Average Glucose Level, and BMI."
    )

elif predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "gender": [gender],

        "age": [age],

        "hypertension": [
            hypertension
        ],

        "heart_disease": [
            heart_disease
        ],

        "ever_married": [
            ever_married
        ],

        "work_type": [
            work_type
        ],

        "Residence_type": [
            residence_type
        ],

        "avg_glucose_level": [
            avg_glucose_level
        ],

        "bmi": [
            bmi
        ],

        "smoking_status": [
            smoking_status
        ]
    })


    try:

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        if preprocessor is not None:

            processed_data = (
                preprocessor.transform(
                    input_data
                )
            )

        else:

            processed_data = input_data


        # ----------------------------------------------------
        # PREDICT PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(
                    processed_data
                )[0][1]
            )

        else:

            prediction_value = (
                model.predict(
                    processed_data
                )[0]
            )

            probability = float(
                prediction_value
            )

       
        # ----------------------------------------------------
        # APPLY OPTIMIZED THRESHOLD
        # ----------------------------------------------------

        prediction = int(
            probability >= threshold
        )
        # ----------------------------------------------------
        # SAVE PREDICTION HISTORY
        # ----------------------------------------------------
        
        history_record = {
            "Date and Time": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "Patient Age": age,
            "Estimated Score": f"{probability * 100:.2f}%",
            "Threshold": f"{threshold:.2f}",
            "Prediction": (
                "Predicted Stroke Risk"
                if prediction == 1
                else "No Stroke Predicted"
                )
        }
        
        st.session_state.prediction_history.append(
            history_record
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown("---")

        st.header(
            "📊 Prediction Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        with result_col1:

            st.metric(
                "Stroke Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2f}"
            )


        # ----------------------------------------------------
        # PROBABILITY INTERPRETATION
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Predicted Stroke Risk"
            )

            st.warning(
                f"""
                The model estimates a stroke
                probability of
                **{probability * 100:.2f}%**.

                This probability is above the
                selected decision threshold of
                **{threshold:.2f}**.
                """
            )

        else:

            st.success(
                "✅ No Stroke Predicted"
            )

            st.info(
                f"""
                The model estimates a stroke
                probability of
                **{probability * 100:.2f}%**.

                This probability is below the
                selected decision threshold of
                **{threshold:.2f}**.
                """
            )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "👤 Patient Information"
        )

        display_data = pd.DataFrame({

            "Parameter": [

                "Age",
                "Gender",
                "Hypertension",
                "Heart Disease",
                "Ever Married",
                "Work Type",
                "Residence Type",
                "Average Glucose Level",
                "BMI",
                "Smoking Status"

            ],

            "Value": [

                age,

                gender,

                "Yes"
                if hypertension == 1
                else "No",

                "Yes"
                if heart_disease == 1
                else "No",

                ever_married,

                work_type,

                residence_type,

                avg_glucose_level,

                bmi,

                smoking_status
            ]
        })
        

        # Streamlit's table renderer may reject mixed numeric/text values
        # in one column. Convert every patient value to text before display.
        display_data["Value"] = display_data["Value"].astype(str)

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )
                # ====================================================
        # PATIENT RISK FACTOR VISUALIZATION
        # ====================================================

        st.markdown("---")
        st.header("📊 Patient Health Factor Dashboard")

        st.write(
            "Visual summary of the health information "
            "entered for this patient."
        )

        # Numeric health information
        factor_col1, factor_col2, factor_col3 = st.columns(3)

        with factor_col1:
            st.metric(
                "Age",
                f"{age:.0f} years"
            )

        with factor_col2:
            st.metric(
                "BMI",
                f"{bmi:.1f}"
            )

        with factor_col3:
            st.metric(
                "Average Glucose",
                f"{avg_glucose_level:.1f}"
            )

        # Health condition indicators
        st.subheader("Health Condition Indicators")

        condition_col1, condition_col2, condition_col3 = st.columns(3)

        with condition_col1:
            st.write("**Hypertension**")
            if hypertension == 1:
                st.warning("Yes")
            else:
                st.success("No")

        with condition_col2:
            st.write("**Heart Disease**")
            if heart_disease == 1:
                st.warning("Yes")
            else:
                st.success("No")

        with condition_col3:
            st.write("**Smoking Status**")
            st.info(smoking_status.title())

        st.caption(
            "These indicators summarize the entered patient "
            "information. They do not independently determine "
            "medical stroke risk."
        )
    
        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown("---")
        st.header("🔍 SHAP - Prediction Explainability")

        st.write(
            "This section explains which patient features "
            "influenced the XGBoost model's output."
        )
        shap_df = None
        try:
            if preprocessor is None:
                raise ValueError(
                    "Preprocessor is required for SHAP explanations."
                )

            # Get names of transformed input features
            feature_names = list(
                preprocessor.get_feature_names_out()
            )

            # Create SHAP Tree Explainer
            explainer = shap.TreeExplainer(model)

            # Calculate SHAP values
            shap_output = explainer.shap_values(
                processed_data
            )

            # Handle different SHAP output formats
            if isinstance(shap_output, list):
                shap_array = np.asarray(
                    shap_output[1]
                    if len(shap_output) > 1
                    else shap_output[0]
                )
            else:
                shap_array = np.asarray(shap_output)

            if shap_array.ndim == 3:
                if shap_array.shape[-1] == 2:
                    shap_array = shap_array[0, :, 1]
                elif shap_array.shape[0] == 2:
                    shap_array = shap_array[1, 0, :]
                else:
                    raise ValueError(
                        "Unexpected SHAP output dimensions."
                    )

            elif shap_array.ndim == 2:
                shap_array = shap_array[0]

            elif shap_array.ndim != 1:
                raise ValueError(
                    "Unsupported SHAP output format."
                )

            # Check feature-name compatibility
            if len(feature_names) != len(shap_array):
                raise ValueError(
                    "SHAP values and feature names do not match."
                )

            # Prepare feature contribution table
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_array
            })

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = shap_df.sort_values(
                by="Absolute Impact",
                ascending=False
            ).head(10)

            # Display top contributing features
            st.subheader("Top 10 Influential Features")

            fig, ax = plt.subplots(figsize=(9, 5))

            shap_colors = [
                 "#D9534F" if value > 0 else "#3984C6"
                for value in shap_df["SHAP Value"]
            ]

            ax.barh(
                shap_df["Feature"],
                shap_df["SHAP Value"],
                color=shap_colors
            )

            ax.axvline(
                0,
                color="black",
                linewidth=0.8
            )

            ax.set_xlabel("SHAP Value (Model Output Impact)")
            ax.set_title("SHAP Feature Contribution")

            ax.invert_yaxis()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Positive SHAP values push the model output "
                "higher, while negative values push it lower. "
                "These values explain model behaviour and do "
                "not establish medical causes."
            )

            st.subheader("SHAP Contribution Details")

            st.dataframe(
                shap_df[
                    ["Feature", "SHAP Value"]
                ].reset_index(drop=True),
                width="stretch",
                hide_index=True
            )

        except Exception as shap_error:
            st.warning(
                f"SHAP explanation could not be generated: "
                f"{shap_error}"
            )
                    # ====================================================
        # DOWNLOADABLE PATIENT PDF REPORT
        # ====================================================

        st.markdown("---")
        st.header("📄 Download Patient Report")

        def generate_patient_report():

            pdf_buffer = io.BytesIO()

            document = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=30,
                leftMargin=30,
                topMargin=25,
                bottomMargin=25
            )

            styles = getSampleStyleSheet()

            styles["Title"].fontSize = 16
            styles["Title"].leading = 19

            styles["Heading2"].fontSize = 11
            styles["Heading2"].leading = 13

            styles["Normal"].fontSize = 8
            styles["Normal"].leading = 10

            content = []

            content.append(
                Paragraph(
                    "STROKE PREDICTION REPORT",
                    styles["Title"]
                )
            )

            content.append(Spacer(1, 8))

            content.append(
                Paragraph(
                    "Patient Information",
                    styles["Heading2"]
                )
            )

            patient_rows = [
                ["Parameter", "Value"],
                ["Age", str(age)],
                ["Gender", gender],
                [
                    "Hypertension",
                    "Yes" if hypertension == 1 else "No"
                ],
                [
                    "Heart Disease",
                    "Yes" if heart_disease == 1 else "No"
                ],
                ["Ever Married", ever_married],
                ["Work Type", work_type],
                ["Residence Type", residence_type],
                ["Average Glucose Level", str(avg_glucose_level)],
                ["BMI", str(bmi)],
                ["Smoking Status", smoking_status]
            ]

            patient_table = Table(
                patient_rows,
                colWidths=[220, 220]
            )

            patient_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ])
            )

            content.append(patient_table)
            content.append(Spacer(1, 28))

            content.append(
                Paragraph(
                    "Prediction Result",
                    styles["Heading2"]
                )
            )

            result_text = (
                "Predicted Stroke Risk"
                if prediction == 1
                else "No Stroke Predicted"
            )

            result_rows = [
                ["Parameter", "Value"],
                ["Prediction", result_text],
                [
                    "Model-estimated score",
                    f"{probability * 100:.2f}%"
                ],
                [
                    "Decision Threshold",
                    f"{threshold:.2f}"
                ]
            ]

            result_table = Table(
                result_rows,
                colWidths=[220, 220]
            )

            result_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ])
            )

            content.append(result_table)

            # Include SHAP details when available
            if shap_df is not None:

                content.append(Spacer(1, 8))

                content.append(
                    Paragraph(
                        "Top SHAP Contributing Features",
                        styles["Heading2"]
                    )
                )

                shap_rows = [["Feature", "SHAP Value"]]

                for _, row in shap_df.head(10).iterrows():

                    shap_rows.append([
                        str(row["Feature"]),
                        f"{row['SHAP Value']:.5f}"
                    ])

                shap_table = Table(
                    shap_rows,
                    colWidths=[300, 140]
                )

                shap_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("PADDING", (0, 0), (-1, -1), 3),
                    ])
                )

                content.append(shap_table)

            content.append(Spacer(1, 10))

            content.append(
                Paragraph(
                    "Disclaimer: This report is generated by an "
                    "academic machine-learning research prototype. "
                    "It is not a medical diagnosis and must not "
                    "replace professional medical advice.",
                    styles["Normal"]
                )
            )

            document.build(content)

            pdf_buffer.seek(0)

            return pdf_buffer.getvalue()

        pdf_data = generate_patient_report()

        st.download_button(
            label="⬇️ Download Patient Report (PDF)",
            data=pdf_data,
            file_name="stroke_prediction_report.pdf",
            mime="application/pdf",
            width="stretch"
        )



    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)

# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown("---")

st.header("📜 Prediction History")

st.write(
    "View the predictions made during the current application session."
)

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )

    st.caption(
        f"Total predictions: {len(history_df)}"
    )

    if st.button(
        "🗑️ Clear Prediction History",
        type="secondary"
    ):
        st.session_state.prediction_history = []
        st.success("Prediction history cleared.")
        st.rerun()

else:

    st.info(
        "No prediction history available. "
        "Make a prediction to see it here."
    )

# ============================================================
# MODEL PERFORMANCE SECTION
# ============================================================

st.markdown("---")

st.header(
    "📈 Model Performance"
)

st.write(
    """
    The following section compares the evaluated machine-learning models.
    XGBoost is used as the final model for patient predictions. Its reported
    performance is based on the held-out test dataset.
    """
)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

comparison_df = None

if os.path.exists(
    COMPARISON_PATH
):

    try:

        comparison_df = pd.read_csv(
            COMPARISON_PATH
        )

    except Exception as e:

        st.warning(
            f"⚠️ Could not load model comparison: {e}"
        )


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

if comparison_df is not None:

    st.subheader(
        "🤖 Machine Learning Model Comparison - Validation Set"
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# FINAL TEST SET COMPARISON: BALANCED RF VS XGBOOST
# ============================================================

st.markdown("---")
st.subheader("🏆 Final Test Set Comparison")

st.write(
    "This comparison uses the independent test dataset. "
    "The thresholds were selected earlier using the validation dataset."
)

test_comparison_df = None

if os.path.exists(TEST_COMPARISON_PATH):

    try:
        test_comparison_df = pd.read_csv(TEST_COMPARISON_PATH)

        required_test_columns = {
            "Model", "Threshold", "Accuracy", "Precision",
            "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC",
            "TN", "FP", "FN", "TP"
        }

        if not required_test_columns.issubset(test_comparison_df.columns):
            st.warning(
                "The test comparison file does not contain all expected columns. "
                "Please run train_model.py again to regenerate it."
            )
            test_comparison_df = None

    except Exception as e:
        st.warning(f"Could not load final test comparison: {e}")

else:
    st.info(
        "Final test comparison file was not found. "
        "Run train_model.py to generate models/test_model_comparison.csv."
    )

if test_comparison_df is not None and not test_comparison_df.empty:

    display_test_comparison = test_comparison_df.copy()

    percentage_columns = [
        "Accuracy", "Precision", "Recall", "Specificity", "F1"
    ]

    for column in percentage_columns:
        display_test_comparison[column] = (
            display_test_comparison[column] * 100
        ).map(lambda value: f"{value:.2f}%")

    for column in ["ROC_AUC", "PR_AUC", "Threshold"]:
        display_test_comparison[column] = (
            display_test_comparison[column].map(lambda value: f"{value:.4f}")
        )

    st.dataframe(
        display_test_comparison,
        width="stretch",
        hide_index=True
    )

    # Visual comparison chart for held-out test metrics.
    st.subheader("📊 Test Metrics Comparison Chart")
    chart_metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    chart_df = test_comparison_df.set_index("Model")[chart_metrics].copy()
    st.bar_chart(chart_df, height=320)
    st.caption("Metric values are proportions from the held-out test set; higher is better.")

    brf_row = test_comparison_df[
        test_comparison_df["Model"].str.contains(
            "Balanced Random Forest", case=False, na=False
        )
    ]

    xgb_row = test_comparison_df[
        test_comparison_df["Model"].str.contains(
            "XGBoost", case=False, na=False
        )
    ]

    if not brf_row.empty and not xgb_row.empty:

        brf_metrics = brf_row.iloc[0]
        xgb_metrics = xgb_row.iloc[0]

        st.subheader("Test Performance at a Glance")

        brf_col, xgb_col = st.columns(2)

        with brf_col:
            st.markdown("**Balanced Random Forest**")
            st.metric("Accuracy", f"{brf_metrics['Accuracy'] * 100:.2f}%")
            st.metric("Sensitivity / Recall", f"{brf_metrics['Recall'] * 100:.2f}%")
            st.metric("F1-score", f"{brf_metrics['F1'] * 100:.2f}%")
            st.metric("ROC-AUC", f"{brf_metrics['ROC_AUC']:.4f}")
            st.caption(f"Decision threshold: {brf_metrics['Threshold']:.2f}")

        with xgb_col:
            st.markdown("**XGBoost**")
            st.metric("Accuracy", f"{xgb_metrics['Accuracy'] * 100:.2f}%")
            st.metric("Sensitivity / Recall", f"{xgb_metrics['Recall'] * 100:.2f}%")
            st.metric("F1-score", f"{xgb_metrics['F1'] * 100:.2f}%")
            st.metric("ROC-AUC", f"{xgb_metrics['ROC_AUC']:.4f}")
            st.caption(f"Decision threshold: {xgb_metrics['Threshold']:.2f}")

        st.subheader("Confusion Matrix Counts")

        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            st.markdown("**Balanced Random Forest**")
            st.dataframe(
                pd.DataFrame(
                    [
                        [int(brf_metrics["TN"]), int(brf_metrics["FP"])],
                        [int(brf_metrics["FN"]), int(brf_metrics["TP"])]
                    ],
                    index=["Actual: No Stroke", "Actual: Stroke"],
                    columns=["Predicted: No Stroke", "Predicted: Stroke"]
                ),
                width="stretch"
            )

        with cm_col2:
            st.markdown("**XGBoost**")
            st.dataframe(
                pd.DataFrame(
                    [
                        [int(xgb_metrics["TN"]), int(xgb_metrics["FP"])],
                        [int(xgb_metrics["FN"]), int(xgb_metrics["TP"])]
                    ],
                    index=["Actual: No Stroke", "Actual: Stroke"],
                    columns=["Predicted: No Stroke", "Predicted: Stroke"]
                ),
                width="stretch"
            )

        st.subheader("Confusion Matrix Visualization")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, row, title in [
            (axes[0], brf_metrics, "Balanced Random Forest"),
            (axes[1], xgb_metrics, "XGBoost - Final Model")
        ]:
            matrix = np.array([
                [int(row["TN"]), int(row["FP"])],
                [int(row["FN"]), int(row["TP"])]
            ])
            image = axis.imshow(matrix, cmap="Blues")
            axis.set_title(title)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_xticks([0, 1], labels=["No Stroke", "Stroke"])
            axis.set_yticks([0, 1], labels=["No Stroke", "Stroke"])
            for r in range(2):
                for c in range(2):
                    axis.text(c, r, str(matrix[r, c]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.success(
            "XGBoost is selected as the final prediction model for this project. "
            "On this test split it achieved higher accuracy, precision and F1-score "
            "than Balanced Random Forest. Balanced Random Forest achieved higher "
            "sensitivity, so the trade-off should be considered when interpreting results."
        )

        st.caption(
            "These are results from one held-out dataset split. "
            "They are experimental model-evaluation results, not evidence of clinical performance."
        )


# ============================================================
# FINAL MODEL METRICS
# ============================================================

st.subheader(
    "🎯 Final XGBoost Performance"
)


if final_metrics is not None:

    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    with metric1:

        st.metric(
            "Accuracy",
            f"{final_metrics['Accuracy'] * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Precision",
            f"{final_metrics['Precision'] * 100:.2f}%"
        )

    with metric3:

        st.metric(
            "Sensitivity",
            f"{final_metrics['Sensitivity'] * 100:.2f}%"
        )


    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    metric4, metric5, metric6 = (
        st.columns(3)
    )

    with metric4:

        st.metric(
            "Specificity",
            f"{final_metrics['Specificity'] * 100:.2f}%"
        )

    with metric5:

        st.metric(
            "F1 Score",
            f"{final_metrics['F1'] * 100:.2f}%"
        )

    with metric6:

        st.metric(
            "ROC-AUC",
            f"{final_metrics['ROC_AUC']:.4f}"
        )


    # --------------------------------------------------------
    # THIRD ROW
    # --------------------------------------------------------

    metric7, metric8 = (
        st.columns(2)
    )

    with metric7:

        st.metric(
            "PR-AUC",
            f"{final_metrics['PR_AUC']:.4f}"
        )

    with metric8:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


else:

    st.warning(
        "⚠️ Final metrics file was not found. "
        "Please run train_model.py first."
    )


# ============================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================

if final_metrics is not None:

    st.markdown("---")

    st.subheader(
        "📋 Final Test Set Summary"
    )

    summary_data = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC",
            "Decision Threshold"
        ],

        "Value": [

            f"{final_metrics['Accuracy'] * 100:.2f}%",

            f"{final_metrics['Precision'] * 100:.2f}%",

            f"{final_metrics['Sensitivity'] * 100:.2f}%",

            f"{final_metrics['Specificity'] * 100:.2f}%",

            f"{final_metrics['F1'] * 100:.2f}%",

            f"{final_metrics['ROC_AUC']:.4f}",

            f"{final_metrics['PR_AUC']:.4f}",

            f"{threshold:.2f}"
        ]
    })

    st.dataframe(
        summary_data,
        width="stretch",
        hide_index=True
    )


# ============================================================
# DECISION THRESHOLD EXPLANATION
# ============================================================

st.markdown("---")

st.header(
    "🎯 Decision Threshold Explanation"
)

st.info(
    f"""
    The XGBoost decision threshold used by the
    application is **{threshold:.2f}**.

    The threshold was selected using the
    validation dataset based on the F1-score.

    If the predicted stroke probability is
    **greater than or equal to {threshold:.2f}**,
    the system displays **Predicted Stroke Risk**.

    If the predicted probability is **less than
    {threshold:.2f}**, the system displays
    **No Stroke Predicted**.
    """
)


# ============================================================
# LOAD THRESHOLD RESULTS
# ============================================================

threshold_df = None

if os.path.exists(
    THRESHOLD_RESULTS_PATH
):

    try:

        threshold_df = pd.read_csv(
            THRESHOLD_RESULTS_PATH
        )

    except Exception:

        threshold_df = None


# ============================================================
# THRESHOLD TABLE
# ============================================================

if threshold_df is not None:

    st.subheader(
        "📊 Threshold Evaluation"
    )

    display_threshold_df = threshold_df.copy()

    display_threshold_df["Selected"] = (
        np.isclose(
            display_threshold_df["Threshold"],
            threshold
        )
    )

    st.dataframe(
        display_threshold_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# THRESHOLD EXAMPLES
# ============================================================

st.subheader(
    "💡 Decision Examples"
)

example_probabilities = [
    0.30,
    0.50,
    threshold - 0.01,
    threshold,
    0.80
]

example_predictions = []

for probability_value in example_probabilities:

    if probability_value >= threshold:

        example_predictions.append(
            "Predicted Stroke Risk"
        )

    else:

        example_predictions.append(
            "No Stroke Predicted"
        )


example_data = pd.DataFrame({

    "Stroke Probability": [
        f"{value * 100:.0f}%"
        for value in example_probabilities
    ],

    "Prediction": example_predictions
})

st.table(
    example_data
)


# ============================================================
# GRAPH SECTION
# ============================================================

st.markdown("---")

st.header(
    "📊 Model Evaluation Graphs"
)

st.write(
    """
    The following graphs were generated during
    model training and final evaluation.
    """
)



# ============================================================
# GRAPH DISPLAY FUNCTION
# ============================================================

def display_graph(
    title,
    image_path,
    description
):
    st.subheader(title)

    st.caption(description)

    if os.path.exists(image_path):

        _, center, _ = st.columns([1, 2, 1])

        with center:
            st.image(
                image_path,
                width=650
            )

    else:
        st.warning(
            f"Graph file not found: "
            f"{os.path.basename(image_path)}"
        )

# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

display_graph(

    "1️⃣ XGBoost Confusion Matrix",

    CONFUSION_MATRIX_PATH,

    """
    The confusion matrix shows correct and incorrect
    XGBoost predictions for stroke and no-stroke classes
    using the validation-selected decision threshold.
    """
)


# ============================================================
# 2. ROC CURVE
# ============================================================

display_graph(

    "2️⃣ ROC Curve Comparison",

    ROC_PATH,

    """
    ROC curves compare the ability of the
    evaluated machine-learning models to
    distinguish between stroke and no-stroke
    cases.
    """
)


# ============================================================
# 3. PRECISION-RECALL CURVE
# ============================================================

display_graph(

    "3️⃣ Precision-Recall Curve",

    PR_PATH,

    """
    The Precision-Recall curve is particularly
    useful for evaluating classification
    performance on an imbalanced dataset.
    """
)


# ============================================================
# 4. THRESHOLD TUNING
# ============================================================

display_graph(

    "4️⃣ Threshold Tuning",

    THRESHOLD_PATH,

    f"""
    This graph shows how Precision, Recall and
    F1-score change across different decision
    thresholds. The selected threshold is
    **{threshold:.2f}**.
    """
)


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

display_graph(

    "5️⃣ XGBoost Feature Importance",

    FEATURE_IMPORTANCE_PATH,

    """
    Feature importance shows the relative
    contribution of the processed input features
    to the XGBoost model.
    """
)


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

if os.path.exists(
    FEATURE_CSV_PATH
):

    st.subheader(
        "🔎 Top Important Features"
    )

    try:

        feature_df = pd.read_csv(
            FEATURE_CSV_PATH
        )

        st.dataframe(
            feature_df.head(15),
            width="stretch",
            hide_index=True
        )

    except Exception as e:

        st.warning(
            f"Could not load feature importance data: {e}"
        )



# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    """
    ⚠️ Important Disclaimer

    This application is developed for academic
    and research purposes. The prediction generated
    by this system should not be considered a
    medical diagnosis or a substitute for professional
    medical advice.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    **Stroke Prediction Using Machine Learning**

    Developed as an academic machine-learning project.
    """
)