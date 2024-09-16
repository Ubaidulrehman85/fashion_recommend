import streamlit as st
import pandas as pd
import joblib
import requests

# Load the trained model and label encoders
model = joblib.load('dress_recommendation_model1.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to make predictions
def predict_dress(gender, age, country, occasion):
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

    # Make a prediction using the loaded model
    predicted_class_encoded = model.predict(input_data)[0]

    # Decode the predicted class back to the original label
    predicted_class = label_encoders['Recommended Dress'].inverse_transform([predicted_class_encoded])[0]
    url = f"https://fashion786.pythonanywhere.com/fashionapp/get-images/?name={predicted_class}"
    response = requests.get(url)
    
    # Extract image paths from the API response
    path = response.json()['data']
    images_path = []
    for path_data in path:
        refine_path = path_data.replace('/home/fashion786/Fashion', '')
        image_url = "https://fashion786.pythonanywhere.com/fashionapp" + refine_path
        images_path.append(image_url)
        
    return images_path

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
    
    # Display the recommended dress images
    if images:
        for img_url in images:
            st.image(img_url, caption="Recommended Dress", width=200)  # Make image larger
    else:
        st.write("No images available.")
