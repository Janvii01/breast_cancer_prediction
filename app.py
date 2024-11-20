import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Streamlit interface
st.title('Breast Cancer Prediction')
st.write("""
This is a simple machine learning app that predicts whether a breast cancer tumor is malignant or benign
based on a variety of features. Enter the values for the features to get a prediction from the model.
""")

# Display a sample of the dataset
st.subheader('Dataset Preview')
st.write(df.head())

# Model Setup
X = df.drop('target', axis=1)
y = df['target']
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
model.fit(X, y)

# User Input Section
st.sidebar.header('Enter Feature Values')

# Use a form to make the inputs clearer and grouped
with st.sidebar.form(key='user_input_form'):
    # Grouped input for all features using sliders
    input_data = [st.slider(feature, min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=0.0) for feature in data.feature_names]
    submit_button = st.form_submit_button(label='Predict')

# Convert user input into a DataFrame
input_df = pd.DataFrame([input_data], columns=data.feature_names)

# Prediction and Output Display
if submit_button:
    prediction = model.predict(input_df)
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    st.subheader(f'Prediction Result: {result}')
    
    # Visual Feedback: Show a bar chart of the input values
    st.subheader('Feature Values')
    input_values = pd.DataFrame(input_data, columns=['Value'], index=data.feature_names)
    st.write(input_values)
    
    # Plot input data
    st.subheader('Input Feature Value Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=input_values.index, y=input_values['Value'], ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)


