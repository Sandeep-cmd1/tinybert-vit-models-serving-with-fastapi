#Streamlit app code using Fast API calls to ML Model
#Run file as 'python -m streamlit run streamlit2_app.py --server.enableXsrfProtection false'
#Above extra line prevents error when uploading files

import streamlit as st
import requests
import json
import os
from scripts import s3_data

Base_API_URL = "http://127.0.0.1:8000/api/v1/"
headers = {
    'Content-Type': 'application/json'
}

st.title("ML Model Served Over REST API in Streamlit Server")

model = st.selectbox("Select Model",["Sentiment Classifier","Disaster Classifier","Pose Classifier"])

if model == "Sentiment Classifier":
    text = st.text_area("Enter your movie review")
    user_id = st.text_input("Enter user id","udemy@kkk.com")

    data = {
        "text":[text],
        "user_id":user_id
    }
    model_api = "sentiment_analysis"

elif model == "Disaster Classifier":
    text = st.text_area("Enter your tweet")
    user_id = st.text_input("Enter user id","udemy@kkk.com")

    data = {
        "text":[text],
        "user_id":user_id
    }
    model_api = "disaster_classifier"

elif model == "Pose Classifier":
    #Image input is of two types here -> 1)Direct online URL 2)Local2S3upload & get URL
    select_file = st.radio("Select image source",["Local","URL"])
    
    if select_file=="URL":
        url = st.text_input("Enter your image URL")

    else:
        image = st.file_uploader("Upload the image",type=["jpg","jpeg","png"])

        file_path = "st_upload_images/temp.jpg"
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        if image is not None:
            with open(file_path,"wb") as f:
                f.write(image.read())
            url = s3_data.upload_image(file_path)

        else:
            url=""

    user_id = st.text_input("Enter user id","udemy@kkk.com")

    data = {
        "url":[url],
        "user_id":user_id
    }
    model_api = "pose_classifier"

if st.button("Predict"):
    with st.spinner("Predicting....Please wait!!!"):
        response = requests.request("POST",Base_API_URL+model_api, headers=headers, data=json.dumps(data))
        output = response.json()
    
    st.write(output)



