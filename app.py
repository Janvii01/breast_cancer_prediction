# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset from sklearn
data = load_breast_cancer()
# Convert to a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
 # Adding the target variable
df['target'] = data.target

# Streamlit interface configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")

# Title and description 
st.title('Breast Cancer Prediction 🎗️')  # Display the main title
st.write("""This is a simple machine learning app that predicts whether a breast cancer tumor is **malignant** or **benign**
            based on a variety of features. Enter the values for the features to get a prediction from the model.""")  # Short description of the app

# Dataset overview section
st.subheader("📊 Dataset Overview")
st.write(df.head())

# Model Setup
X = df.drop('target', axis=1)
y = df['target']
# Defining a neural network model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
# Training the model on the dataset
model.fit(X, y)

# Sidebar for user input
st.sidebar.header('📝 Enter Feature Values')

# Using a form to make the inputs clearer and grouped
with st.sidebar.form(key='user_input_form'):
    st.write("Adjust the sliders to input feature values")
    # Creating sliders for each feature
    input_data = [st.slider(feature, min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=0.0, step=0.01)
                  for feature in data.feature_names]
    # Adding a button to submit the form
    submit_button = st.form_submit_button(label='Predict')

# Convert user input into a DataFrame
input_df = pd.DataFrame([input_data], columns=data.feature_names)

# Initialize result to a placeholder
result = None

# Prediction and Output Display
if submit_button:   # Check if the user submitted the form
    with st.spinner('Making prediction...'):  # Showing a spinner while predicting
        prediction = model.predict(input_df)  # Make a prediction based on user input
        result = 'Malignant 🩸' if prediction[0] == 0 else 'Benign 🟢'
    
    # Display the prediction result
    st.subheader(f'🔮 **Prediction Result:** {result}')
    
    # Show the input values in a more readable format
    st.subheader('🧑‍🔬 Feature Values')
    input_values = pd.DataFrame(input_data, columns=['Value'], index=data.feature_names)
    st.write(input_values)

    # Plot the user input data
    st.subheader('📈 Input Feature Value Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=input_values.index, y=input_values['Value'], ax=ax, palette="viridis")
    plt.xticks(rotation=90)

    # Set the y-axis ticks to have a step of 10
    ax.set_yticks(range(0, int(input_values['Value'].max()) + 10, 10))

    # Display the plot
    st.pyplot(fig)

# Add a conclusion message
st.write("""
    Thank you for using the Breast Cancer Prediction app! 🎉
    Based on the input data, our machine learning model has predicted the tumor as **{}**.
    Keep learning, stay healthy, and remember, early detection saves lives! 🚀
    """.format(result if result is not None else "no prediction yet"))
