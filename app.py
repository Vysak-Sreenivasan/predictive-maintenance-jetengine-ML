import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load your trained model
pick = pickle.load(open('JetEngineConditionPredictionModel.pkl', 'rb'))

# Title and description
st.title("Jet Engine Condition Prediction Model")
st.markdown("""
This tool predicts the probability of Jet Engine Condition based on operational parameters. 
Enter the parameters details below to get real-time predictions.
""")

# Background Image URL (replace with your image URL)
background_image_url = "https://epiphanyinc.net/wp-content/uploads/2024/03/Aviation-software.png"  # Replace this with the actual image URL

# Add background image to Streamlit app using custom CSS
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url({background_image_url}) no-repeat center center fixed;
        background-size: cover;
        color: white;
    }}
    .sidebar .sidebar-content {{
        background: rgba(0, 0, 0, 0.6);
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Input fields with corrected initial values within valid ranges
st.header("Input Parameters")
time_cycles = st.number_input("Operating Cycle", min_value=1.00, max_value=550.00, step=0.1, value=1.0)
setting_1 = st.number_input("Operational setting -1 of the engine", min_value=0.0, max_value=50.0, step=0.1, value=0.0)
setting_2 = st.number_input("Operational setting -2 of the engine", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
setting_3 = st.number_input("Operational setting -3 of the engine", min_value=60.0, max_value=100.0, step=0.1, value=60.0)
T2 = st.number_input("Total temperature at fan inlet (°R)", min_value=445.0, max_value=520.0, step=0.1, value=445.0)
T24 = st.number_input("Total temperature at LPC outlet (°R)", min_value=530.0, max_value=650.0, step=0.1, value=535.0)
T30 = st.number_input("Total temperature at HPC outlet (°R)", min_value=1240.0, max_value=1620.0, step=0.1, value=1242.7)
T50 = st.number_input("Total temperature at LPT outlet (°R)", min_value=1020.0, max_value=1440.0, step=0.1, value=1023.8)
P2 = st.number_input("Pressure at fan inlet (psia)", min_value=4.0, max_value=14.5, step=0.1, value=4.0)  # Set to min_value
P15 = st.number_input("Total pressure in bypass-duct (psia)", min_value=5.5, max_value=21.5, step=0.1, value=5.7)
P30 = st.number_input("Total pressure at HPC outlet (psia)", min_value=136.0, max_value=570.0, step=0.1, value=136.2)
Nf = st.number_input("Physical fan speed (rpm)", min_value=1900.0, max_value=2400.0, step=50.0, value=1914.7)
Nc = st.number_input("Physical core speed (rpm)", min_value=7980.0, max_value=9250.0, step=50.0, value=7984.5)
epr = st.number_input("Engine pressure ratio (P50/P2)", min_value=0.93, max_value=1.33, step=0.01, value=0.93)
Ps30 = st.number_input("Static pressure at HPC outlet (psia)", min_value=36.0, max_value=49.0, step=0.1, value=36.0)
phi = st.number_input("Ratio of fuel flow to Ps30 (pps/psi)", min_value=128.0, max_value=540.0, step=0.1, value=128.3)
NRf = st.number_input("Corrected fan speed (rpm)", min_value=2025.0, max_value=2390.0, step=50.0, value=2027.6)
NRc = st.number_input("Corrected core speed (rpm)", min_value=7850.0, max_value=8300.0, step=50.0, value=7850.0)
BPR = st.number_input("Bypass Ratio", min_value=8.0, max_value=11.1, step=0.1, value=8.16)
htBleed = st.number_input("Bleed Enthalpy", min_value=300.0, max_value=400.0, step=0.1, value=302.0)
Nf_dmd = st.number_input("Demanded fan speed (rpm)", min_value=1915.0, max_value=2390.0, step=50.0, value=1915.0)
PCNfR_dmd = st.number_input("Demanded corrected fan speed (rpm)", min_value=85.0, max_value=100.0, step=0.1, value=85.0)  # Set to min_value
W31 = st.number_input("HPT coolant bleed (lbm/s)", min_value=10.0, max_value=40.0, step=0.1, value=10.16)
W32 = st.number_input("LPT coolant bleed (lbm/s)", min_value=6.0, max_value=24.0, step=0.1, value=6.01)

# Prepare data for prediction
input_data = pd.DataFrame({
    'time_cycles': [time_cycles],
    'setting_1': [setting_1],
    'setting_2': [setting_2],
    'setting_3': [setting_3],
    'T2': [T2],
    'T24': [T24],
    'T30': [T30],
    'T50': [T50],
    'P2': [P2],
    'P15': [P15],
    'P30': [P30],
    'Nf': [Nf],
    'Nc': [Nc],
    'epr': [epr],
    'Ps30': [Ps30],
    'phi': [phi],
    'NRf': [NRf],
    'NRc': [NRc],
    'BPR': [BPR],
    'htBleed': [htBleed],
    'Nf_dmd': [Nf_dmd],
    'PCNfR_dmd': [PCNfR_dmd],
    'W31': [W31],
    'W32': [W32]
})


# Prediction
if st.button("Predict Failure"):
    input_df = pd.DataFrame(input_data)
    scaled_data = pick['scaler'].transform(input_df)
    prediction = pick['model'].predict(input_df)[0]

    # Display Prediction Results
    st.subheader("Prediction Results")
    if prediction == 0:
        st.success("""The engine is in Good Condition. 
        The Life Ratio indicates minimal wear, and no immediate maintenance is required. The engine is operating well within safe parameters.""")
    elif prediction == 1:
        st.warning("""The engine is in Moderate Condition.
        The Life Ratio suggests that the engine has undergone some wear but remains functional. Routine maintenance is advisable to ensure continued performance.""")
    else:
        st.error("The engine is in Warning Condition. The Life Ratio is close to the end-of-life threshold, indicating significant wear. Immediate maintenance is recommended to prevent potential failure.")

    # Feature Visualization
    st.subheader("Input Parameter Visualization")
    fig, ax = plt.subplots()
    ax.barh(input_data.columns, input_data.iloc[0], color="skyblue")
    ax.set_xlabel("Value")
    ax.set_title("Input Feature Values")
    st.pyplot(fig)

    # Download Option
    st.download_button(
        label="Download Prediction Result",
        data=input_data.to_csv(index=False),
        file_name="prediction_results.csv",
        mime="text/csv"
    )

# Model Description (Optional)
with st.expander("About the Model"):
    st.write(
        "This model uses sensor data to predict engine failure probabilities. The model was trained on historical data to identify patterns associated with failures."
    )
