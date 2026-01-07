
#Import Data Models (I/P,O/P) for ML model API calls from other code
from scripts.data_model import NLPDataInput,NLPDataOutput,ImageDataInput,ImageDataOutput

#Call on AWS S3 data downloading code 
from scripts import s3_data

#ML model handling imports
from transformers import pipeline
import torch

#Only Image processing pipeline needs image_processor parameter too
from transformers import AutoImageProcessor
model_chkpt = 'google/vit-base-patch16-224-in21k'
image_processor = AutoImageProcessor.from_pretrained(model_chkpt,use_fast=True)

#Required imports for code below
from fastapi import FastAPI
from fastapi import Request
import uvicorn
import os
import time

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

######Downloading all ML models & Create Hugging Face pipeline using ML model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

force_download = False #Keep True until full download happens, next change to False

model_name = 'tinybert-sentimet-classifier/'
local_path='ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3_data.download_directory(local_path,model_name)
sentiment_model = pipeline('text-classification',model=local_path,device=device)

model_name = 'tinybert-disaster-tweet-classifier/'
local_path='ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3_data.download_directory(local_path,model_name)
tweets_model = pipeline('text-classification',model=local_path,device=device)

model_name = 'vit-human-pose-classification/'
local_path='ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3_data.download_directory(local_path,model_name)
pose_model = pipeline('image-classification',model=local_path,device=device,image_processor=image_processor)

#Downloading all ML models & creation of their pipelines ENDS here  


######FAST API Code starts here 
# Downloading & loading model are kept outside of FastAPI to prevent hectic loading for every API call

app = FastAPI()

#GET method with root need to check server running or not (GET  brings msg to confirm)
@app.get("/")
def read_root():
    return "Hello, I am working!"

#POST method ONLY can use Data Models

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data:NLPDataInput): #NLPDataInput is a Pydantic Data Model

    #Add NLPDataOutput items in code and return it as O/P
    start = time.time() #Find time taken by classifier 
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000) 

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    final_output = NLPDataOutput(model_name="tinybert-sentimet-classifier",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)

    return final_output

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data:NLPDataInput):

    #Add NLPDataOutput items in code and return it as O/P
    start = time.time() #Find time taken by classifier 
    output = tweets_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000) 

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    final_output = NLPDataOutput(model_name="tinybert-disaster-tweet-classifier",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)

    return final_output

@app.post("/api/v1/pose_classifier")
def pose_classifier(data:ImageDataInput):

    #Add ImageDataOutput items in code and return it as O/P
    start = time.time() #Find time taken by classifier 
    image_urls = [str(url) for url in data.url] #ML model takes list[str] not rigid HttpUrl pydantic objects in List, so convert them
    output = pose_model(image_urls)
    end = time.time()
    prediction_time = int((end-start)*1000) 

    labels = [x[0]['label'] for x in output]
    scores = [x[0]['score'] for x in output]

    final_output = ImageDataOutput(model_name="vit-human-pose-classification",
                           url=data.url,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)

    return final_output



#Below lines autorun the above code without need for terminal run 'python -m uvicorn fastapi1:app --reload'
#Just run this file directly by play button or run 'python file.py' in terminal, no need mention uvicorn
if __name__=="__main__":
    uvicorn.run(app="fastapi_main:app", port=8000, reload=True)     #port is optional,default is 8000
