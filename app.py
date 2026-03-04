import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder  


import pickle
import streamlit as st

#LOad the model
model=tf.keras.models.load_model("model.h5")
with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

st.title("Diabetes Prediction")

#User Inputs

Pregnancies=st.slider("Pregnancies",0,20)
Glucose=st.number_input("Glucose")
BloodPressure=st.number_input("BloodPressure")
SkinThickness=st.number_input("SkinThickness")
Insulin=st.number_input("Insulin")
BMI=st.number_input("BMI",min_value=20.0,max_value=70.0,step=0.1)
DiabetesPedigreeFunction=st.number_input("DiabetesPedigreeFunction")
Age=st.slider("Age",0,100)


input_data=pd.DataFrame({
    "Pregnancies":[Pregnancies],
    "Glucose":[Glucose],
    "BloodPressure":[BloodPressure],
    "SkinThickness":[SkinThickness],
    "Insulin":[Insulin],
    "BMI":[BMI],
    "DiabetesPedigreeFunction":[DiabetesPedigreeFunction],
    "Age":[Age],
})


input_scaled=scaler.transform(input_data)

prediction=model.predict(input_scaled)

if prediction[0][0] > 0.5:
    st.error("Person is likely to have diabetes")
else:
    st.success("Person is unlikely to have diabetes")
