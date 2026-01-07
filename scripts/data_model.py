#Design pydantic data models to accept I/Ps (params/ data payload) and O/Ps (score, class, latency)
#Pydantic data models are ONLY for POST method

from pydantic import BaseModel
from pydantic import EmailStr, HttpUrl

class NLPDataInput(BaseModel):
    text:list[str]
    user_id:EmailStr

class ImageDataInput(BaseModel):
    url:list[HttpUrl]
    user_id:EmailStr

class NLPDataOutput(BaseModel):
    model_name:str
    text:list[str]
    labels:list[str]
    scores:list[float]
    prediction_time:int

class ImageDataOutput(BaseModel):
    model_name:str
    url:list[HttpUrl]
    labels:list[str]
    scores:list[float]
    prediction_time:int