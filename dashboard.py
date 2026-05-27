import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

st.title("AI Education System")

df = pd.read_csv("cleaned_data.csv")

study_hours=st.slider("Select Study Hours",0,20,2)
sleep_hours=st.slider("Sleep Hours",0,20,4)
stress_level=st.selectbox("Choose Stress Level",("Low","Moderate","High"))
physical_activity=st.slider("Physical Activity",0,10,1)

le = LabelEncoder()

df['stress_encoded'] = le.fit_transform(df['stress_level'])
stress_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

stress_encoded = stress_map[stress_level]


input_df=pd.DataFrame({
    'study_hours_per_day':[study_hours],
    'sleep_hours_per_day':[sleep_hours],
    'stress_encoded':[stress_encoded],
    'physical_activity_hours_per_day':[physical_activity]
})

X = df[['study_hours_per_day',
        'sleep_hours_per_day',
        'stress_encoded',
        'physical_activity_hours_per_day']]

Y = df['gpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=200
)

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

if st.button("Predict"):
    prediction =regr.predict(input_df)
    st.success(f"Predicted GPA: {prediction[0]:.2f}")
