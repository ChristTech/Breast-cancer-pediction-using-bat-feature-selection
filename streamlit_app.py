import streamlit as st
import pandas as pd
import joblib

# Load the saved model
# Ensure the model file 'svm_model.pkl' is in the same directory as your app
try:
    svm_model = joblib.load('svm_model.pkl')
except FileNotFoundError:
    st.error("Model file 'svm_model.pkl' not found. Please make sure it's in the correct directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="♋",
    layout="wide",
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Custom CSS for background and card styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1579546929518-9e396f3cc809?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
    }
    .main-card {
        background-color: rgba(255, 255, 255, 0.7); /* White with 70% opacity */
        padding: 25px;
        border-radius: 15px;
    }
    .result-card {
        background-color: rgba(255, 255, 255, 0.8); /* White with 80% opacity */
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def show_input_page():
    """Displays the page for user to input patient data."""
    st.title('Breast Cancer Diagnosis Prediction')
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.header('Enter Patient Data')

        col1, col2, col3 = st.columns(3)
        with col1:
            radius1 = st.number_input('Radius (mean)', min_value=6.0, max_value=30.0, value=14.1, format="%.3f")
            texture1 = st.number_input('Texture (mean)', min_value=9.0, max_value=40.0, value=19.3, format="%.3f")
            perimeter1 = st.number_input('Perimeter (mean)', min_value=40.0, max_value=190.0, value=92.0, format="%.3f")
            symmetry1 = st.number_input('Symmetry (mean)', min_value=0.1, max_value=0.4, value=0.18, format="%.3f")

        with col2:
            radius2 = st.number_input('Radius (se)', min_value=0.1, max_value=3.0, value=0.4, format="%.3f")
            smoothness2 = st.number_input('Smoothness (se)', min_value=0.001, max_value=0.04, value=0.007, format="%.4f")
            concavity2 = st.number_input('Concavity (se)', min_value=0.0, max_value=0.4, value=0.025, format="%.4f")
            symmetry2 = st.number_input('Symmetry (se)', min_value=0.007, max_value=0.08, value=0.02, format="%.4f")

        with col3:
            radius3 = st.number_input('Radius (worst)', min_value=7.0, max_value=40.0, value=16.3, format="%.3f")
            texture3 = st.number_input('Texture (worst)', min_value=12.0, max_value=50.0, value=25.7, format="%.3f")
            perimeter3 = st.number_input('Perimeter (worst)', min_value=50.0, max_value=260.0, value=107.3, format="%.3f")
            compactness3 = st.number_input('Compactness (worst)', min_value=0.02, max_value=1.1, value=0.25, format="%.3f")
            concavity3 = st.number_input('Concavity (worst)', min_value=0.0, max_value=1.3, value=0.27, format="%.3f")
            concave_points3 = st.number_input('Concave Points (worst)', min_value=0.0, max_value=0.3, value=0.11, format="%.3f")

        if st.button('Predict Diagnosis', key='predict_button'):
            user_data = {
                'radius1': radius1, 'texture1': texture1, 'perimeter1': perimeter1, 'symmetry1': symmetry1,
                'radius2': radius2, 'smoothness2': smoothness2, 'concavity2': concavity2, 'symmetry2': symmetry2,
                'radius3': radius3, 'texture3': texture3, 'perimeter3': perimeter3, 'compactness3': compactness3,
                'concavity3': concavity3, 'concave_points3': concave_points3
            }
            features = pd.DataFrame(user_data, index=[0])
            prediction = svm_model.predict(features)
            st.session_state.prediction_result = prediction[0]
            st.session_state.page = 'result'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def show_result_page():
    """Displays the prediction result with a corresponding image."""
    result = st.session_state.prediction_result
    st.title('Diagnosis Result')

    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if result == 1:
            st.error('Diagnosis: Malignant')
            st.image("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144", caption="Malignant cells under a microscope.", use_container_width=True)
            st.warning("The model predicts a high probability of malignancy. Please consult a medical professional for a definitive diagnosis and further action.")
        else:
            st.success('Diagnosis: Benign')
            st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef", caption="Consulting with a healthcare professional.", use_container_width=True)
            st.info("The model predicts a high probability of a benign condition. However, always follow up with a healthcare provider for confirmation.")

        if st.button('Make Another Diagnosis', key='back_button'):
            st.session_state.page = 'input'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Main app router
if st.session_state.page == 'input':
    show_input_page()
else:
    show_result_page()