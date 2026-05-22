import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model('best_mlp.keras')
        scaler = joblib.load('scaler.joblib')
        return model, scaler, True, None
    except Exception as e:
        return None, None, False, str(e)

model, scaler, model_loaded, load_error = load_model_and_scaler()

# Main UI
st.markdown("# 🏥 Diabetes Risk Prediction System")
st.markdown("**Advanced Deep Learning Model for Early Diabetes Detection**")

if not model_loaded:
    st.error(f"❌ Error loading model or scaler: {load_error}")
    st.stop()

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Analytics", "ℹ️ Information", "📜 History"])

with tab1:
    st.markdown("## Enter Patient Information")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Basic Information")
        pregnancies = st.slider(
            'Number of Pregnancies',
            min_value=0, max_value=17, value=3,
            help="Number of times pregnant"
        )

        age = st.slider(
            'Age (years)',
            min_value=21, max_value=81, value=33,
            help="Age of the patient in years"
        )

        diabetes_pedigree_function = st.slider(
            'Diabetes Pedigree Function',
            min_value=0.078, max_value=2.42, value=0.471,
            step=0.01,
            help="Genetic diabetes risk (higher = more family history)"
        )

    with col2:
        st.markdown("### Health Metrics")
        glucose = st.slider(
            'Glucose (mg/dL)',
            min_value=0, max_value=200, value=120,
            help="Fasting blood glucose level"
        )

        blood_pressure = st.slider(
            'Blood Pressure (mmHg)',
            min_value=0, max_value=122, value=70,
            help="Diastolic blood pressure"
        )

        bmi = st.slider(
            'BMI (kg/m²)',
            min_value=0.0, max_value=67.1, value=32.0,
            step=0.1,
            help="Body Mass Index"
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Advanced Metrics")
        skin_thickness = st.slider(
            'Skin Thickness (mm)',
            min_value=0, max_value=99, value=20,
            help="Triceps skin fold thickness"
        )

    with col4:
        insulin = st.slider(
            'Insulin (mu U/ml)',
            min_value=0, max_value=846, value=79,
            help="2-hour serum insulin level"
        )

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    )

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Prediction button
    col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 2])

    with col_pred2:
        if st.button('🔮 Predict', key='predict_btn', use_container_width=True):
            with st.spinner('Analyzing patient data...'):
                prediction_proba = model.predict(input_data_scaled, verbose=0)[0][0]
                prediction = (prediction_proba >= 0.5).astype(int)

            # Store in history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data': input_data.to_dict('records')[0],
                'probability': float(prediction_proba),
                'prediction': int(prediction)
            })

            # Display results
            st.markdown("---")
            st.markdown("## 📋 Prediction Results")

            # Determine risk level
            if prediction_proba >= 0.7:
                risk_level = "🔴 HIGH RISK"
                risk_class = "risk-high"
                risk_color = "#f44336"
            elif prediction_proba >= 0.4:
                risk_level = "🟡 MODERATE RISK"
                risk_class = "risk-medium"
                risk_color = "#ff9800"
            else:
                risk_level = "🟢 LOW RISK"
                risk_class = "risk-low"
                risk_color = "#4caf50"

            # Create metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    label="Risk Assessment",
                    value=risk_level,
                    delta=f"{prediction_proba*100:.1f}% Probability"
                )

            with metric_col2:
                st.metric(
                    label="Prediction",
                    value="Diabetes Likely" if prediction == 1 else "No Diabetes",
                    delta="Positive" if prediction == 1 else "Negative"
                )

            with metric_col3:
                st.metric(
                    label="Confidence",
                    value=f"{abs(prediction_proba - 0.5) * 200:.1f}%"
                )

            # Display detailed probability
            st.markdown("### Probability Distribution")
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Diabetes', 'Diabetes'],
                    y=[1-prediction_proba, prediction_proba],
                    marker=dict(color=['#4caf50', '#f44336']),
                    text=[f'{(1-prediction_proba)*100:.1f}%', f'{prediction_proba*100:.1f}%'],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Prediction Probability",
                yaxis_title="Probability",
                xaxis_title="Outcome",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display gauge chart
            st.markdown("### Risk Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Risk (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#e8f5e9"},
                        {'range': [40, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Health recommendations
            st.markdown("### 💡 Health Recommendations")
            recommendations = []

            if glucose > 126:
                recommendations.append("⚠️ **High Fasting Glucose**: Consider consulting an endocrinologist for glucose management")
            elif glucose < 70:
                recommendations.append("⚠️ **Low Glucose**: Monitor for hypoglycemia symptoms")

            if bmi >= 30:
                recommendations.append("📉 **Overweight/Obese**: Weight management through diet and exercise is recommended")

            if blood_pressure >= 140 or blood_pressure <= 60:
                recommendations.append("❤️ **Abnormal Blood Pressure**: Monitor regularly and consult a physician")

            if insulin > 200:
                recommendations.append("🔬 **High Insulin Level**: Possible insulin resistance; dietary modifications may help")

            if diabetes_pedigree_function > 1.0:
                recommendations.append("👨‍👩‍👧 **Strong Family History**: Preventive measures are crucial")

            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ All metrics are within reasonable ranges. Continue with regular check-ups.")

with tab2:
    st.markdown("## 📊 Analytics Dashboard")

    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", len(history_df))

        with col2:
            positive = (history_df['prediction'] == 1).sum()
            st.metric("Positive Cases", positive)

        with col3:
            avg_prob = history_df['probability'].mean()
            st.metric("Average Risk Probability", f"{avg_prob:.2%}")

        with col4:
            max_prob = history_df['probability'].max()
            st.metric("Highest Risk Score", f"{max_prob:.2%}")

        # Prediction trend
        st.markdown("### Prediction Trend")
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        fig = px.line(
            history_df,
            x='timestamp',
            y='probability',
            markers=True,
            title="Diabetes Risk Probability Over Time",
            labels={'probability': 'Risk Probability', 'timestamp': 'Date/Time'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk distribution
        st.markdown("### Risk Distribution")
        risk_counts = history_df['prediction'].value_counts().sort_index()
        # Create proper labels based on what exists in the data
        labels = {0: 'No Diabetes', 1: 'Diabetes'}
        names = [labels[idx] for idx in risk_counts.index]
        colors = {0: '#4caf50', 1: '#f44336'}
        colors_list = [colors[idx] for idx in risk_counts.index]
        
        fig = px.pie(
            values=risk_counts.values,
            names=names,
            title="Distribution of Predictions",
            color_discrete_sequence=colors_list
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature analysis
        st.markdown("### Feature Trends")
        feature = st.selectbox(
            "Select feature to analyze",
            ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'Age']
        )

        # Extract features from the 'data' column dictionary
        feature_data = history_df['data'].apply(lambda x: x.get(feature) if isinstance(x, dict) else None)
        analysis_df = history_df.copy()
        analysis_df[feature] = feature_data
        
        fig = px.scatter(
            analysis_df,
            x=feature,
            y='probability',
            color='prediction',
            title=f"{feature} vs Diabetes Risk",
            labels={'probability': 'Risk Probability', 'prediction': 'Outcome'},
            color_discrete_map={0: '#4caf50', 1: '#f44336'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.markdown("### Prediction History Table")
        display_df = history_df.copy()
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
        display_df['prediction'] = display_df['prediction'].apply(lambda x: "Diabetes" if x == 1 else "No Diabetes")
        st.dataframe(display_df, use_container_width=True)

    else:
        st.info("No predictions yet. Make a prediction in the 'Prediction' tab to see them here.")

with tab3:
    st.markdown("## ℹ️ About the Model & Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Information")
        st.markdown("""
        - **Model Type**: Deep Learning Multi-Layer Perceptron (MLP)
        - **Framework**: TensorFlow/Keras
        - **Purpose**: Early diabetes risk detection
        - **Input Features**: 8 patient health metrics
        - **Output**: Binary classification (Diabetes: Yes/No)
        """)

    with col2:
        st.markdown("### Dataset & Training")
        st.markdown("""
        - **Training Data**: Diabetes prediction dataset
        - **Training Samples**: Thousands of patient records
        - **Model Validation**: Cross-validation with train-test split
        - **Scaling**: StandardScaler applied to all features
        """)

    st.markdown("---")
    st.markdown("### Feature Descriptions")

    features_info = {
        "Pregnancies": {
            "Description": "Number of times the patient has been pregnant",
            "Range": "0-17",
            "Unit": "count",
            "Clinical Relevance": "Gestational diabetes history can indicate future diabetes risk"
        },
        "Glucose": {
            "Description": "Fasting blood glucose concentration (2 hours after last meal)",
            "Range": "0-200",
            "Unit": "mg/dL",
            "Clinical Relevance": "Normal: <100 mg/dL, Prediabetes: 100-125 mg/dL, Diabetes: ≥126 mg/dL"
        },
        "Blood Pressure": {
            "Description": "Diastolic blood pressure (pressure between heartbeats)",
            "Range": "0-122",
            "Unit": "mmHg",
            "Clinical Relevance": "Hypertension increases diabetes risk"
        },
        "Skin Thickness": {
            "Description": "Triceps skin fold thickness (measure of body fat)",
            "Range": "0-99",
            "Unit": "mm",
            "Clinical Relevance": "Correlates with insulin resistance"
        },
        "Insulin": {
            "Description": "2-hour serum insulin level (fasting test)",
            "Range": "0-846",
            "Unit": "mu U/ml",
            "Clinical Relevance": "High insulin indicates possible insulin resistance"
        },
        "BMI": {
            "Description": "Body Mass Index (weight relative to height)",
            "Range": "0-67.1",
            "Unit": "kg/m²",
            "Clinical Relevance": "Normal: <25, Overweight: 25-29.9, Obese: ≥30"
        },
        "Diabetes Pedigree Function": {
            "Description": "Genetic predisposition to diabetes based on family history",
            "Range": "0.078-2.42",
            "Unit": "score",
            "Clinical Relevance": "Higher scores indicate stronger family history of diabetes"
        },
        "Age": {
            "Description": "Age of the patient at the time of examination",
            "Range": "21-81",
            "Unit": "years",
            "Clinical Relevance": "Diabetes risk increases with age"
        }
    }

    for feature, info in features_info.items():
        with st.expander(f"📌 {feature}"):
            st.markdown(f"""
            **Description**: {info['Description']}

            **Range**: {info['Range']} {info['Unit']}

            **Clinical Relevance**: {info['Clinical Relevance']}
            """)

    st.markdown("---")
    st.markdown("### Disclaimer")
    st.warning("""
    ⚠️ **Important Notice**: This application is designed for educational and informational purposes only. 
    The predictions made by this model should NOT be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. 
    The model's predictions are based on statistical patterns and may not be 100% accurate for all individuals.
    """)

with tab4:
    st.markdown("## 📜 Prediction History")

    if st.session_state.prediction_history:
        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.success("History cleared!")
                st.rerun()

        history_df = pd.DataFrame(st.session_state.prediction_history)

        # Display history with better formatting
        for idx, record in enumerate(reversed(st.session_state.prediction_history), 1):
            with st.container():
                col_time, col_risk, col_details = st.columns([2, 2, 1])

                with col_time:
                    st.markdown(f"**{record['timestamp']}**")

                with col_risk:
                    if record['prediction'] == 1:
                        st.markdown("🔴 **Diabetes Likely**")
                    else:
                        st.markdown("🟢 **No Diabetes**")

                with col_details:
                    st.markdown(f"**{record['probability']*100:.1f}%** risk")

                # Expandable details
                with st.expander("View Details", key=f"expand_{idx}"):
                    detail_cols = st.columns(4)
                    data = record['data']

                    with detail_cols[0]:
                        st.metric("Glucose", f"{data['Glucose']:.0f} mg/dL")
                        st.metric("BMI", f"{data['BMI']:.1f} kg/m²")

                    with detail_cols[1]:
                        st.metric("Blood Pressure", f"{data['BloodPressure']:.0f} mmHg")
                        st.metric("Age", f"{data['Age']:.0f} years")

                    with detail_cols[2]:
                        st.metric("Pregnancies", f"{data['Pregnancies']:.0f}")
                        st.metric("Insulin", f"{data['Insulin']:.0f} mu U/ml")

                    with detail_cols[3]:
                        st.metric("Skin Thickness", f"{data['SkinThickness']:.0f} mm")
                        st.metric("DPF", f"{data['DiabetesPedigreeFunction']:.2f}")

                st.markdown("---")

        # Export history
        st.markdown("### Export History")

        col_export1, col_export2 = st.columns(2)

        with col_export1:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col_export2:
            json_data = json.dumps(st.session_state.prediction_history, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    else:
        st.info("No prediction history yet. Make predictions in the 'Prediction' tab to see them here.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>Diabetes Risk Prediction System v1.0 | Powered by TensorFlow & Streamlit</p>
    <p>For medical concerns, always consult with healthcare professionals</p>
</div>
""", unsafe_allow_html=True)

