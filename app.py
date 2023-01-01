import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pickled model
file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Load data used for training
data = pd.read_csv("traineddata.csv")

st.title("Laptop Price Predictors")

# Rest of your code...

# Inside the prediction button click block
if st.button('Predict Price'):
    try:
        ppi = None
        # Convert 'Yes'/'No' to 1/0
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        X_resolution = int(resolution.split('x')[0])
        Y_resolution = int(resolution.split('x')[1])

        ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size

        # Ensure the order of features matches the training data
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        # Reshape the query array to match the expected shape
        query = query.reshape(1, -1)

        # Make the prediction
        prediction = int(np.exp(rf.predict(query)[0]))

        st.title("Predicted price for this laptop could be between " +
                 str(prediction - 1000) + "₹" + " to " + str(prediction + 1000) + "₹")

    except Exception as e:
        st.error("An error occurred during prediction: " + str(e))
