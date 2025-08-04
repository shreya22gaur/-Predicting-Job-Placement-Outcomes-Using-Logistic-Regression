# app.py

import numpy as np
import streamlit as st
import pickle
from PIL import Image

# Load trained model
with open('placement.pkl', 'rb') as f:
    lg = pickle.load(f)

# UI
st.title("🎓 Job Placement Prediction")
img = Image.open("Job-Placement-Agency.jpg")
st.image(img, width=650)

st.subheader("Enter the 14 feature values below (comma-separated):")
example = "77.0,87.0,59.0,68.0,68.63,0,0,0,1,0,0,1,0,0"
input_text = st.text_input("Example Input:", example)

if input_text:
    try:
        input_list = list(map(float, input_text.split(',')))
        if len(input_list) != 14:
            st.error("Please enter exactly 14 comma-separated values.")
        else:
            np_df = np.array(input_list).reshape(1, -1)
            prediction = lg.predict(np_df)
            if prediction[0] == 1:
                st.success("✅ This person is likely to be Placed!")
            else:
                st.error("❌ This person is NOT likely to be Placed.")
    except ValueError:
        st.error("Please enter only numeric values.")

