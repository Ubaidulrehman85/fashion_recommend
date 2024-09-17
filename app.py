import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np
import webbrowser

# Load the trained model and label encoders
model = joblib.load('dress_recommendation_model1.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to make predictions
def predict_dress(gender, age, country, occasion):
    try:
        # Encode user inputs using the same label encoders as in training
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        country_encoded = label_encoders['Country'].transform([country])[0]
        occasion_encoded = label_encoders['Occasion'].transform([occasion])[0]

        # Prepare the data for prediction
        input_data = pd.DataFrame({
            'Gender': [gender_encoded],
            'Age': [age],
            'Country': [country_encoded],
            'Occasion': [occasion_encoded]
        })

        # Check for missing values
        if input_data.isnull().values.any():
            st.error("Input contains missing values. Please check the input fields.")
            return None

        # Make a prediction using the loaded model
        predicted_class_encoded = model.predict(input_data)[0]

        # Decode the predicted class back to the original label
        predicted_class = label_encoders['Recommended Dress'].inverse_transform([predicted_class_encoded])[0]
        
        # Fetch image URLs from external API
        url = f"https://fashion786.pythonanywhere.com/fashionapp/get-images/?name={predicted_class}"
        response = requests.get(url)
        if response.status_code == 200:
            path = response.json().get('data', [])
            images_path = []
            for path_data in path:
                refine_path = path_data.replace('/home/fashion786/Fashion', '')
                image_url = "https://fashion786.pythonanywhere.com/fashionapp" + refine_path
                images_path.append(image_url)
            return images_path
        else:
            st.error("Failed to fetch images from API.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit UI components
st.title("Dress Recommendation System")
st.write("Provide your details to get a dress recommendation.")

# Get the choices for the dropdowns from the label encoders
gender_options = list(label_encoders['Gender'].classes_)
country_options = list(label_encoders['Country'].classes_)
occasion_options = list(label_encoders['Occasion'].classes_)

# User input components
gender = st.selectbox("Gender", gender_options)
age = st.slider("Age", min_value=18, max_value=60, value=25)
country = st.selectbox("Country", country_options)
occasion = st.selectbox("Occasion", occasion_options)

# When the user clicks the 'Recommend' button, make the prediction
if st.button("Recommend"):
    images = predict_dress(gender, age, country, occasion)
    
    # Display the recommended dress images in a gallery view
    if images:
        cols = st.columns(3)  # Create 3 columns for the gallery view
        for idx, img_url in enumerate(images):
            with cols[idx % 3]:  # Cycle through columns
                st.image(img_url, caption="Recommended Dress", width=200)
    else:
        st.write("No images available or an error occurred.") 

# "Shop Now" button
URL_STRING = "http://localhost/weiboo/PHP/index-two.php"
st.markdown(
    f'<div style="position: absolute; top: 10px; right: 0px;">'
    f'<a href="{URL_STRING}" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Shop Now</a>'
    f'</div>',
    unsafe_allow_html=True
)
