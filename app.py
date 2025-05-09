import streamlit as st
import mlflow
from auth import show_login_signup, logout
import prediction_file  # Replace with your actual prediction logic
from upload_data import load_and_process_data  # Replace with actual function
from what_if import what_if_analysis  # Replace with actual function
from data_preview import show_data_preview  # Replace with actual function

st.set_page_config(page_title="Predictik - Time Series Forecasting", page_icon="📊")

# 🚨 Check if the user is authenticated (and avoid blocking after login)
if "user" not in st.session_state or not st.session_state['authenticated']:
    # If the user is not logged in, show the login/signup page
    show_login_signup()
    st.stop()  # 🔐 Block the app unless logged in

with st.sidebar:
    st.markdown(f"👤 Logged in as: `{st.session_state['user']['email']}`")
    if st.button("Logout"):
        logout()         # updates state
        st.rerun()       # rerun from top-level only

 # Rerun the app after logout

mlflow.set_tracking_uri("mlruns")
st.title("Predictik - Time Series Forecasting")
st.sidebar.success("Sélectionnez une page ci-dessus.")
st.markdown(
    """
    ### 📈 Upload historical stock data to generate predictions
    - 🔍 **Analyse** historical data for trends
    - 📊 **Identify** patterns and correlations
    - 🤖 **Use** machine learning for forecasting
    - 💰 **Make** informed investment decisions
    """
)

# Sidebar navigation for different pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Upload Data", "Prediction", "Data Preview", "What-If Analysis"])

# Initialize session state variables if not already present
if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None
if 'selected_id' not in st.session_state:
    st.session_state['selected_id'] = None
if 'df_grouped_historical' not in st.session_state:
    st.session_state['df_grouped_historical'] = None

# Conditional rendering based on the page the user selects
if page == "Upload Data":
    st.subheader("Upload Your Sales Data (ZIP containing a single CSV)")
    uploaded_df = load_and_process_data()  # Call your upload function here
    if uploaded_df is not None:
        st.session_state['uploaded_df'] = uploaded_df
        st.write("Uploaded Data Sample:")
        st.dataframe(uploaded_df.head())
    else:
        st.error("No data uploaded. Please upload a valid ZIP file containing a single CSV file.")

elif page == "Prediction":
    if st.session_state.get('uploaded_df') is not None:
        prediction_file.prediction()  # Replace with your actual prediction function
    else:
        st.info("Please upload data on the 'Upload Data' page first.")

elif page == "Data Preview":
    if st.session_state.get('uploaded_df') is not None:
        show_data_preview()  # Replace with your actual data preview function
    else:
        st.info("Please upload data on the 'Upload Data' page first.")

elif page == "What-If Analysis":
    if st.session_state.get('uploaded_df') is not None:
        what_if_analysis()  # Replace with your actual What-If analysis function
    else:
        st.info("Please upload data on the 'Upload Data' page first.")
