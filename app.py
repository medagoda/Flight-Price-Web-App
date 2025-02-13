from datetime import datetime
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import load_model

# Load the trained model
model = load_model()

# Streamlit UI
st.title("Flight Price Prediction")

page_bg_img = '''
<style>
.stApp {
background-image: url("https://img.freepik.com/premium-photo/airliner-airport-runway-apron_1417-7394.jpg?ga=GA1.1.1792054496.1731576720&semt=ais_hybrid");
background-size: cover;
background-position: center;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# User Inputs
departure_date = st.date_input("Departure Date", min_value=datetime.today())
arrival_date = st.date_input("Arrival Date", min_value=departure_date)
source = st.selectbox("Source", ["Chennai", "Delhi", "Kolkata", "Mumbai"])
destination = st.selectbox("Destination", ["Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"])
stoppage = st.selectbox("Stoppage Type", ["Non-stop", "1 Stop", "2 Stops", "3 Stops", "4 Stops"])
airline = st.selectbox("Airline", ["Air India", "GoAir", "IndiGo", "Jet Airways", "Jet Airways Business",
                                  "Multiple carriers", "Multiple carriers Premium economy",
                                  "SpiceJet", "Trujet", "Vistara", "Vistara Premium economy"])

# Convert inputs to a DataFrame
input_data = pd.DataFrame({
    "Departure Date": [departure_date.strftime("%Y-%m-%d")],
    "Arrival Date": [arrival_date.strftime("%Y-%m-%d")],
    "Source": [source],
    "Destination": [destination],
    "Stoppage": [stoppage],
    "Airline": [airline]
})

# Fit and transform the categorical columns (e.g., Source, Destination, Airline, Stoppage)
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Encoding the categorical features (Source, Destination, Airline, Stoppage)
input_data_encoded = encoder.fit_transform(input_data[["Source", "Destination", "Airline", "Stoppage"]])

# Get the column names of the encoded columns
encoded_columns = encoder.get_feature_names_out(["Source", "Destination", "Airline", "Stoppage"])

# Convert the encoded data into a DataFrame
encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns)

# Combine the encoded columns with the rest of the input data (i.e., dates)
input_data_final = pd.concat([input_data.drop(columns=["Source", "Destination", "Airline", "Stoppage"]), encoded_df], axis=1)

# Ensure the final dataset has the same columns as the training dataset
# Get the feature names from the model and align columns in the input data
model_feature_names = model.feature_names_in_  # Model's expected features

# Add missing columns if any (set to 0)
missing_cols = set(model_feature_names) - set(input_data_final.columns)
for col in missing_cols:
    input_data_final[col] = 0

# Reorder columns to match the model's feature order
input_data_final = input_data_final[model_feature_names]

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data_final)
    st.success(f"Predicted Flight Price: ${prediction[0]:,.2f}")
