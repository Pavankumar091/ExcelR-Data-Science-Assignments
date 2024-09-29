# import streamlit as st
# import joblib
# import numpy as np

# # Load the trained logistic regression model
# model = joblib.load('logistic_regression_model.pkl')

# st.title('Logistic Regression Model Deployment')

# # Create inputs for user to enter feature values
# st.header('Input Features')
# # Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
# feature1 = st.number_input('Pclass')
# feature2 = st.number_input('Sex')
# feature3 = st.number_input('Age')
# feature4 = st.number_input('SibSp')
# feature5 = st.number_input('Parch')
# feature6 = st.number_input('Fare')
# feature7 = st.number_input('Embarked')
# # Add more inputs as necessary

# # Make prediction
# if st.button('Predict'):
#     features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])  # Add more features as necessary
#     prediction = model.predict(features)
#     probability = model.predict_proba(features)
    
#     st.write(f'Prediction: {prediction[0]}')
#     st.write(f'Probability: {probability[0]}')

# st.write("This is a simple logistic regression model deployed using Streamlit.")


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained logistic regression model
model = joblib.load('logistic_regression_model.pkl')

# Features used in the model
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Convert categorical columns to appropriate format
def preprocess_input(input_features):
    df = pd.DataFrame([input_features])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    return df

st.title('Logistic Regression Model Deployment')

# Create inputs for user to enter feature values
st.header('Input Features')

input_features = {}
input_features['Pclass'] = st.selectbox('Pclass', [1, 2, 3])
input_features['Sex'] = st.selectbox('Sex', ['male', 'female'])
input_features['Age'] = st.number_input('Age', min_value=0.42, max_value=80.0, value=29.0)
input_features['SibSp'] = st.number_input('SibSp', min_value=0, max_value=8, value=0)
input_features['Parch'] = st.number_input('Parch', min_value=0, max_value=6, value=0)
input_features['Fare'] = st.number_input('Fare', min_value=0.0, value=32.0)
input_features['Embarked'] = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Preprocess the input features
features_df = preprocess_input(input_features)

# Ensure the order of columns matches the training data
features_df = features_df[feature_names]

# Make prediction
if st.button('Predict'):
    features = np.array(features_df)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    st.write(f'Prediction: {prediction[0]}')
    st.write(f'Probability: {probability[0]}')

st.write("This is a simple logistic regression model deployed using Streamlit.")
