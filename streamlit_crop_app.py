import mysql.connector
import pickle
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
import hmac

# Authentication
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text("Crop Recommendation and Advisory System!")
            st.text("Please Enter Your Credentials")
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()

# Main Streamlit app starts here

# Load the data
data_csv = pd.read_csv("Crop_recommendation.csv")
grouped = data_csv.groupby(by='label').mean().reset_index()
crop_labels = data_csv['label'].unique()

# Load the models and encoders
rf_classifier = pickle.load(open("rf_classifier_model.pkl", 'rb'))
xgb_classifier = pickle.load(open("xgb_classifier_model.pkl", 'rb'))
encoder = pickle.load(open("label_encoder.pkl", 'rb'))
models = [rf_classifier, xgb_classifier]

st.set_page_config(
    page_title = 'Crop Recommendation and Advisor',
    page_icon = '✅',
    layout = 'wide'
    )

# Summary statistics for each crop
def summary(crop):
    x = data_csv[data['label'] == crop]
    nitrogen, phosp = st.columns(2)
    with nitrogen:
        st.subheader("Nitrogen")
        st.write("- Minimum Nitrogen required:", x['N'].min())
        st.write("- Average Nitrogen required:", x['N'].mean())
        st.write("- Maximum Nitrogen required:", x['N'].max())
    with phosp:
        st.subheader("Phosphorus")
        st.write("- Minimum Phosphorus required:", x['P'].min())
        st.write("- Average Phosphorus required:", x['P'].mean())
        st.write("- Maximum Phosphorus required:", x['P'].max())
    st.write("---------------------------------------------")
    potas, temp = st.columns(2)
    with potas:
        st.subheader("Potassium")
        st.write("- Minimum Potassium required:", x['K'].min())
        st.write("- Average Potassium required:", x['K'].mean())
        st.write("- Maximum Potassium required:", x['K'].max())
    with temp:
        st.subheader("Temperature")
        st.write("- Minimum Temperature required: {:.2f}".format(x['temperature'].min()))
        st.write("- Average Temperature required: {:.2f}".format(x['temperature'].mean()))
        st.write("- Maximum Temperature required: {:.2f}".format(x['temperature'].max()))
    st.write("---------------------------------------------")
    hum, pH = st.columns(2)
    with hum:
        st.subheader("Humidity")
        st.write("- Minimum Humidity required: {:.2f}".format(x['humidity'].min()))
        st.write("- Average Humidity required: {:.2f}".format(x['humidity'].mean()))
        st.write("- Maximum Humidity required: {:.2f}".format(x['humidity'].max()))
    with pH:
        st.subheader("pH")
        st.write("- Minimum pH required: {:.2f}".format(x['ph'].min()))
        st.write("- Average pH required: {:.2f}".format(x['ph'].mean()))
        st.write("- Maximum pH required: {:.2f}".format(x['ph'].max()))
    st.write("---------------------------------------------")
    st.subheader("Rainfall")
    st.write("- Minimum Rainfall required: {:.2f}".format(x['rainfall'].min()))
    st.write("- Average Rainfall required: {:.2f}".format(x['rainfall'].mean()))
    st.write("- Maximum Rainfall required: {:.2f}".format(x['rainfall'].max()))


# Visualization of the crop requirements
def viz():
    fig,ax=plt.subplots(7,1,figsize=(25,25))
    for index,i in enumerate(grouped.columns[1:]):
        sns.barplot(data=grouped,x='label',y=i,ax=ax[index])
        plt.suptitle("Comparision of Mean Attributes of various classes",size=25)
        plt.xlabel("")
        
    return st.write(fig)
 
# Prediction function   
def predict_ensemble(data_csv):
    # Preprocess the data
    X = np.array([data_csv])
    
    # Make predictions using each model
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Ensemble by averaging predictions
    ensemble_predictions = sum(predictions) / len(models)
    
    # Round ensemble predictions to the nearest integer 
    ensemble_predictions = ensemble_predictions.round().astype(int)
    
    # Decode the predictions
    decoded_predictions = encoder.inverse_transform(ensemble_predictions)
    
    return decoded_predictions[0]

# Function to get user input and pass it to predictions function
def get_user_input():
    n = st.number_input("Ratio of Nitrogen in Soil:", min_value=20.0, max_value=140.0)
    p = st.number_input("Ratio of Phosphorus in Soil:", min_value=15.0, max_value=145.0)
    k = st.number_input("Ratio of Potassium in Soil:", min_value=150.0, max_value=205.0)
    temperature = st.number_input("Temperature(°C):", min_value=6.0, max_value=43.7)
    humidity = st.number_input("Relative humdity in %:", min_value=40.0, max_value=100.0)
    ph = st.number_input("pH value of the Soil:", min_value=1.0, max_value=14.0)
    rainfall = st.number_input("Rainfall in mm:", min_value=10.0, max_value=300.0)

    # Combine input values into a Python list
    input_data = [n, p, k, temperature, humidity, ph, rainfall]
    return input_data

# Function to read the contents of the Jupyter Notebook file
def read_notebook_file(file_path):
    with open(file_path, "r") as f:
        notebook_content = f.read()
    return notebook_content

notebook_file_path = "crop-recommedation-agri-project.ipynb"  

# Read the contents of the notebook file
notebook_content = read_notebook_file(notebook_file_path)

# Function to interact with ChatGPT
def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
        )
    
    return response.choices[0].message["content"]
    
def create_prompt(crop):
    prompt = f"""
    Optimal farming conditions and best practices for the crop delimited by 
    triple backticks ```{crop}```
    """
    return prompt

def create_promptt(crop):
    prompt = f"""
    possible crop disease outbreaks for the crop delimited by 
    triple backticks and advise on control measures ```{crop}```
    """
    return prompt

def run():
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Home", "Summary Statistics", "Exploratory Data Analysis", "Prediction", "Q&A Section", "Reports"])
    
    if page == "Home":
        st.title("Crop Recommendation")
        st.header('Dataset Overview')
        head, stats = st.columns(2)
        
        with head:
            st.write(data_csv.head(8))
            
        with stats:
            st.write(data_csv.describe())
    
    elif page == "Summary Statistics":
        st.title("Summary statistics for each crop")
        selected_crop = st.selectbox("Select a crop:", crop_labels)
        summary(selected_crop)
        
    elif page == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        viz()
        
        st.subheader('Observations')
        st.markdown('''
                - Cotton requires most Nitrogen.
                - Apple requires most Phosphorus.
                - Grapes require most Potassium
                - Papaya requires a hot climate.
                - Coconut requires a humid climate.
                - Chickpea requires high pH in soil.
                - Rice requires huge amount of Rainfall.
                ''')
        
    elif page == "Prediction":
        st.title("Model Prediction")
        # Get user input
        input_data = get_user_input()

        # Make predictions using the input data
        if st.button("Predict"):
            prediction = predict_ensemble(input_data)
            st.write("The Suggested Crop for the Given Climatic Conditions is: ", prediction)
            prompt = create_prompt(prediction)
            st.subheader("Find Out More About: ", prediction)
            st.write(get_completion(prompt))
            page = st.sidebar.radio("Extras", ["Possible Diseases"])

            if page:
                promptt = create_promptt(prediction)
                st.title('Possible Diseases and Mitigation Measures')
                st.write(get_completion(promptt))

    elif page == "Q&A Section":
        st.title("Question & Answer Section")
        prompt = st.text_area("For all your crop related questions")

        if st.button("Submit"):
            if prompt:
                response = get_completion(prompt)
                st.write(response)

    elif page == "Reports":
        st.title("Soil Report")
        
        def create_connection():
            try:
                connection = mysql.connector.connect(
                    host=os.getenv("34.122.18.217"),
                    user=os.getenv("canvas-hook-425400-s0:us-central1:crop-recommendation-final"),
                    password=os.getenv("12345678"),
                    database=os.getenv("crop-reco"),
                    port=int(os.getenv("DB_PORT", 1433))  # Default port is 3306 if not specified
                )
                return connection
            except mysql.connector.Error as err:
                st.error(f"Error: Could not connect to MySQL server. {err}")
                return None

        # Function to fetch data from the database
        def fetch_data():
            connection = create_connection()
            if connection is None:
                return None
        
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM CropFeatures")
                rows = cursor.fetchall()
                cursor.close()
                connection.close()
                return rows
            except mysql.connector.Error as err:
                st.error(f"Error: {err}")
                return None
        
        # Function to insert data into the database
        def insert_data(feature, value):
            connection = create_connection()
            if connection is None:
                return
        
            try:
                cursor = connection.cursor()
                cursor.execute("INSERT INTO CropFeatures (Feature, Value) VALUES (%s, %s)", (feature, value))
                connection.commit()
                cursor.close()
                connection.close()
                st.success(f"Inserted {feature}: {value} into the database")
            except mysql.connector.Error as err:
                st.error(f"Error: {err}")
        
        # Form to insert new data
        with st.form(key='insert_form'):
            feature = st.text_input("Feature")
            value = st.text_input("Value")
            submit_button = st.form_submit_button(label='Insert')
        
            if submit_button:
                if feature and value:
                    insert_data(feature, value)
                else:
                    st.error("Both fields are required")
        
        # Fetch and display data
        data = fetch_data()
        
        if data:
            st.subheader("Existing Crop Features")
            for row in data:
                st.write(f"**Feature**: {row[0]}")
                st.write(f"**Value**: {row[1]}")
                st.write("---")
        else:
            st.write("No data found or an error occurred.")

if __name__ == '__main__':
    run()
