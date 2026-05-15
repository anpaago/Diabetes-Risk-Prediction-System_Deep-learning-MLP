# 🏥 Diabetes Risk Prediction System

A sophisticated deep learning application for early diabetes detection using a Multi-Layer Perceptron (MLP) neural network. Built with TensorFlow and deployed with a rich, interactive Streamlit UI featuring advanced visualizations and health analytics.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [File Descriptions](#file-descriptions)
- [Screenshots](#screenshots)
- [Health Recommendations](#health-recommendations)
- [Important Disclaimer](#important-disclaimer)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## 🎯 Project Overview

This project implements an advanced diabetes prediction system using deep learning. The system analyzes 8 patient health metrics and uses a trained MLP model to predict the likelihood of diabetes. The application provides:

- **Real-time Predictions**: Instant diabetes risk assessment based on patient input
- **Risk Visualization**: Interactive gauges, charts, and probability distributions
- **Analytics Dashboard**: Track prediction trends and analyze patterns
- **Prediction History**: Complete history of all predictions with export options
- **Health Recommendations**: Personalized health guidance based on input metrics
- **Professional UI**: Modern, responsive interface built with Streamlit

## ✨ Features

### 🔍 Prediction Tab
- **Interactive Input Form**: User-friendly sliders for all 8 health metrics
- **Color-Coded Layout**: Organized by Basic Information, Health Metrics, and Advanced Metrics
- **Real-Time Analysis**: Instant feedback as you adjust parameters
- **Risk Assessment**: Three-tier risk classification (Low/Moderate/High)
- **Probability Visualization**: Bar charts showing prediction breakdown
- **Risk Gauge**: Interactive gauge chart showing risk probability
- **Health Recommendations**: Personalized suggestions based on input values
- **Confidence Metrics**: Display model confidence in predictions

### 📊 Analytics Dashboard
- **Summary Statistics**: Total predictions, positive cases, average risk scores
- **Trend Analysis**: Line charts showing prediction probability over time
- **Risk Distribution**: Pie charts of prediction outcomes
- **Feature Correlation**: Analyze relationships between any feature and diabetes risk
- **Prediction History Table**: Complete record of all predictions

### ℹ️ Information Tab
- **Model Details**: Information about the deep learning model architecture
- **Dataset Information**: Training data statistics and validation methods
- **Feature Descriptions**: Detailed explanations of each health metric
- **Clinical Relevance**: Normal ranges and clinical significance for each feature
- **Medical Disclaimer**: Important legal notice about medical use

### 📜 History Tab
- **Prediction Log**: Complete history with timestamps
- **Expandable Details**: Drill down into individual prediction details
- **Export Options**: Download predictions as CSV or JSON
- **History Management**: Clear history when needed

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Deep Learning** | TensorFlow/Keras | ≥2.13.0 |
| **UI Framework** | Streamlit | ≥1.28.0 |
| **Data Processing** | Pandas | ≥2.0.0 |
| **Numerical Computing** | NumPy | ≥1.24.0 |
| **Machine Learning** | scikit-learn | ≥1.3.0 |
| **Visualization** | Plotly | ≥5.17.0 |
| **Additional** | Matplotlib, Seaborn, joblib |  |
| **Environment** | Python 3.9+ | Virtual Environment |

## 📁 Project Structure

```
Diabetes_DL_MLP/
├── README.md                 # This file - Complete project documentation
├── streamlit_app.py          # Main Streamlit application with rich UI
├── app.py                    # Original simple Streamlit app (legacy)
├── best_mlp.keras            # Trained deep learning model
├── scaler.joblib             # Fitted StandardScaler for feature normalization
├── requirements.txt          # Python dependencies
└── venv/                     # Python virtual environment
    ├── Scripts/
    │   ├── python            # Python interpreter
    │   ├── pip               # Package manager
    │   └── streamlit         # Streamlit CLI
    └── lib/
        └── site-packages/    # Installed packages
```

## 🚀 Installation

## 🧠 FastAPI (API) + Docker Support

This project also includes a FastAPI service that exposes your trained model.

### Run the API (local, without Docker)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API:
   ```bash
   uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
   ```
3. Open Swagger UI at:
   - http://localhost:8000/docs

### Test the API endpoint
Example:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.471,
    "Age": 33
  }'
```

### Run the API with Docker
1. Build:
   ```bash
   docker build -t diabetes-api .
   ```
2. Run:
   ```bash
   docker run -p 8000:8000 diabetes-api
   ```
3. Open:
   - http://localhost:8000/docs

### Docker Compose (optional)
```bash
docker compose up --build
```

---


### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- 500MB+ free disk space

### Step 1: Clone/Download the Project

```bash
cd c:\Users\Paago Anthony\Desktop\Diabetes_DL_MLP
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The installation will download and install:
- streamlit (UI framework)
- tensorflow (deep learning)
- pandas (data manipulation)
- plotly (interactive visualizations)
- scikit-learn (machine learning tools)
- And other required packages

Installation typically takes 5-15 minutes depending on internet speed and system specifications.

## 🎮 Usage

### Running the Application

**Using Virtual Environment (Recommended):**

```powershell
cd "c:\Users\Paago Anthony\Desktop\Diabetes_DL_MLP"
.\venv\Scripts\streamlit run streamlit_app.py
```

**Alternative (Direct with venv):**
```bash
.\venv\Scripts\streamlit run streamlit_app.py
```

The application will start on:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

### Using the Application

#### 1. **Making a Prediction**
   - Navigate to the **🔍 Prediction** tab
   - Adjust sliders for each health metric
   - Click **🔮 Predict** button
   - View results with risk assessment and visualizations

#### 2. **Understanding Results**
   - **Risk Level**: Green (Low), Orange (Moderate), Red (High)
   - **Probability**: Percentage likelihood of diabetes
   - **Recommendations**: Personalized health guidance

#### 3. **Analyzing Trends**
   - Go to **📊 Analytics** tab
   - View prediction history and trends
   - Analyze feature correlations
   - Export data for further analysis

#### 4. **Learning More**
   - Visit **ℹ️ Information** tab
   - Read detailed feature descriptions
   - Understand clinical relevance
   - Review important medical disclaimer

#### 5. **Tracking History**
   - Check **📜 History** tab
   - Review past predictions
   - Export as CSV or JSON
   - Clear history when needed

## 🧠 Model Information

### Architecture
- **Type**: Multi-Layer Perceptron (MLP) Neural Network
- **Framework**: TensorFlow/Keras
- **Input Layer**: 8 features
- **Hidden Layers**: Fully connected layers with activation functions
- **Output Layer**: Binary classification (Sigmoid activation)
- **Loss Function**: Binary crossentropy

### Input Features (8 metrics)

| # | Feature | Range | Unit | Meaning |
|---|---------|-------|------|---------|
| 1 | Pregnancies | 0-17 | count | Number of pregnancies |
| 2 | Glucose | 0-200 | mg/dL | Fasting blood glucose |
| 3 | Blood Pressure | 0-122 | mmHg | Diastolic BP |
| 4 | Skin Thickness | 0-99 | mm | Triceps skin fold |
| 5 | Insulin | 0-846 | mu U/ml | 2-hour serum insulin |
| 6 | BMI | 0-67.1 | kg/m² | Body Mass Index |
| 7 | Diabetes Pedigree Function | 0.078-2.42 | score | Genetic predisposition |
| 8 | Age | 21-81 | years | Patient age |

### Output
- **Binary Classification**: 
  - 0 = No Diabetes
  - 1 = Diabetes Likely
- **Confidence Score**: 0-1 probability

### Data Preprocessing
- **Scaling**: StandardScaler (fitted on training data)
- **Missing Values**: Handled using mean/median imputation
- **Feature Normalization**: Applied before model inference

### Model Performance
- **Training Data**: Thousands of patient records
- **Validation**: Cross-validation with train-test split
- **Task**: Binary classification for diabetes prediction

## 📄 File Descriptions

### streamlit_app.py (Main Application - 350+ lines)
**Purpose**: Rich, interactive Streamlit UI for diabetes prediction

**Key Components**:
- Session state management for prediction history
- 4-tab interface (Prediction, Analytics, Information, History)
- Interactive visualizations using Plotly
- Custom CSS styling for professional appearance
- Model and scaler caching for performance
- Export functionality (CSV/JSON)

**Functions**:
- `load_model_and_scaler()`: Loads pre-trained model and scaler with error handling
- Prediction analysis with risk assessment
- Health recommendation engine
- Analytics dashboard with trend analysis

### app.py (Legacy Application)
**Purpose**: Original simple Streamlit application

**Status**: Kept for reference; use `streamlit_app.py` instead

**Note**: Can be removed if not needed

### best_mlp.keras
**Type**: Trained neural network model (Keras format)
**Size**: ~50-200KB
**Format**: HDF5 (.keras extension)
**Training**: Pre-trained on diabetes prediction dataset
**Framework**: TensorFlow 2.x compatible

### scaler.joblib
**Type**: Fitted StandardScaler object
**Size**: ~1-5KB
**Format**: joblib serialized object
**Purpose**: Feature normalization for model input
**Usage**: Scales new predictions using fitted mean/std from training data

### requirements.txt
**Purpose**: Lists all Python dependencies

**Contents**:
- streamlit≥1.28.0
- tensorflow≥2.13.0
- pandas≥2.0.0
- numpy≥1.24.0
- scikit-learn≥1.3.0
- matplotlib≥3.7.0
- seaborn≥0.12.0
- imbalanced-learn≥0.11.0
- joblib≥1.3.0
- plotly≥5.17.0

### venv/
**Purpose**: Python virtual environment
**Status**: Created during setup
**Size**: ~500MB+ (depending on packages)
**Use**: Isolates project dependencies from system Python

## 🖼️ Screenshots & UI Overview

### Tab 1: Prediction Interface
- Two-column slider layout
- Color-coded sections
- Predict button
- Risk gauge visualization
- Probability distribution chart
- Personalized health recommendations

### Tab 2: Analytics Dashboard
- Summary statistics
- Trend line charts
- Risk distribution pie charts
- Feature correlation scatter plots
- Prediction history table

### Tab 3: Information
- Model and dataset information
- Feature descriptions with expandable cards
- Clinical relevance for each metric
- Medical disclaimer notice

### Tab 4: History
- Timestamped prediction records
- Expandable detail cards
- CSV/JSON export buttons
- Clear history option

## 💡 Health Recommendations

The application provides personalized recommendations based on:

- **Glucose Levels**: 
  - Normal: <100 mg/dL
  - Prediabetes: 100-125 mg/dL
  - Diabetes: ≥126 mg/dL

- **BMI Categories**:
  - Underweight: <18.5
  - Normal: 18.5-24.9
  - Overweight: 25-29.9
  - Obese: ≥30

- **Blood Pressure**:
  - Normal: <120/80 mmHg
  - Elevated: 120-129/<80 mmHg
  - High BP: ≥130/80 mmHg

- **Other Factors**:
  - Insulin resistance indicators
  - Family history (Diabetes Pedigree)
  - Age-related risk factors

## ⚠️ Important Disclaimer

**CRITICAL NOTICE**: 

This application is designed for **educational and informational purposes only**. 

🚨 **DO NOT use this as a substitute for professional medical advice, diagnosis, or treatment.**

- The predictions are based on statistical patterns and machine learning models
- Results may not be 100% accurate for all individuals
- Always consult qualified healthcare professionals for medical decisions
- This system should complement, not replace, medical expertise
- In case of medical emergencies, seek immediate professional medical attention

**Liability**: The developers are not responsible for any medical decisions made based on this application's predictions.

## 🔧 Troubleshooting

### Issue: Module not found errors
**Solution**: 
```bash
pip install -r requirements.txt
```
Ensure virtual environment is activated.

### Issue: Model or scaler file not found
**Solution**: 
- Verify `best_mlp.keras` and `scaler.joblib` exist in the project directory
- Check file paths in `streamlit_app.py`

### Issue: Port 8501 already in use
**Solution**: 
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Slow predictions on first run
**Solution**: 
TensorFlow initialization on first run takes longer. Subsequent predictions will be faster.

### Issue: Virtual environment not activating
**Solution**:
- Check PowerShell execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Use Command Prompt instead: `venv\Scripts\activate`

### Issue: Large file downloads during installation
**Solution**: 
- TensorFlow and other packages are large (~500MB total)
- Ensure stable internet connection
- Installation may take 10-15 minutes on slower connections

## 🚀 Future Enhancements

Potential improvements for future versions:

- [ ] Multi-model comparison (ensemble predictions)
- [ ] User authentication and cloud storage
- [ ] Automated email reports
- [ ] Integration with medical record systems
- [ ] Mobile app version
- [ ] Real-time data sync
- [ ] Advanced statistical analysis
- [ ] Prediction confidence intervals
- [ ] Model retraining pipeline
- [ ] Docker containerization
- [ ] API endpoint deployment
- [ ] Multi-language support
- [ ] Dark mode UI theme
- [ ] Batch prediction processing
- [ ] Integration with wearable devices

## 📞 Support & Contact

For issues or questions:
1. Check the Troubleshooting section
2. Review the Information tab in the app
3. Check project files and dependencies
4. Verify Python and package versions

## 📚 References & Resources

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Diabetes Risk Factors - Mayo Clinic](https://www.mayoclinic.org/)

## 📝 License

This project is provided as-is for educational and research purposes.

## ✅ Verification Checklist

Before deploying or sharing, ensure:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows all required packages)
- [ ] `best_mlp.keras` model file exists
- [ ] `scaler.joblib` scaler file exists
- [ ] `streamlit_app.py` launches without errors
- [ ] All tabs (Prediction, Analytics, Information, History) work correctly
- [ ] Predictions generate and display correctly
- [ ] Charts and visualizations render properly
- [ ] Export functionality works (CSV/JSON)
- [ ] History persistence works across sessions
- [ ] No missing dependencies or import errors

---

**Last Updated**: May 15, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
