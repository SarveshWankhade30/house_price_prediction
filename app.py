# app.py

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
# Collect user inputs for model features
import streamlit as st
import pickle
import pandas as pd

# Load the trained model (which includes the preprocessor)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("House Price Prediction App")

# User inputs for house features
Num_Bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
Square_Footage = st.number_input("Square Footage", min_value=0, value=2000)
Lot_Size = st.number_input("Lot Size (in square feet)", min_value=0, value=0)
Num_Bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=2)
Garage_Size = st.number_input("Garage Size", min_value=0, value=2)
Neighborhood_Quality = st.number_input("Neighborhood Quality", min_value=0, value=1)  # Assuming categorical
Year_Built = st.number_input("Year Built", min_value=1800, value=2000)

# Prepare input data with the exact columns required by the model
input_data = pd.DataFrame({
    'Num_Bedrooms': [Num_Bedrooms],
    'Square_Footage': [Square_Footage],
    'Lot_Size': [Lot_Size],
    'Num_Bathrooms': [Num_Bathrooms],
    'Garage_Size': [Garage_Size],
    'Neighborhood_Quality': [Neighborhood_Quality],
    'Year_Built': [Year_Built]
})

# Predict the house price
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.write(f"Estimated House Price: ${prediction[0]:,.2f}")
