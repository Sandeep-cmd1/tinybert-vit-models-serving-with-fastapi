#Code to download ML models from AWS S3 to here

import boto3
import os

s3 = boto3.client('s3',region_name="ap-south-1") #region_name must for correct presigned_URL generation
bucket_name = 'mlopssandy'

#Assign local_path (in main code) -> local folder path to download from S3
#Define s3_prefix (in main code) -> target S3 folder to download data from

def download_directory(local_path,model_name):
    s3_prefix = 'ml-models/'+model_name
    os.makedirs(local_path,exist_ok=True) 
    paginator = s3.get_paginator('list_objects_v2') 
    for result in paginator.paginate(Bucket=bucket_name,Prefix=s3_prefix):
        if 'Contents' in result:    #'Contents' contain details of all objects including folder paths
            for key in result['Contents']: #Loop through each object
                s3_key = key['Key']   #Extract object folder path in S3
                #Below line copies S3 sub-folder path of each file/obj into local directory
                local_file = os.path.join(local_path,os.path.relpath(s3_key,s3_prefix)).replace('\\','/')  
                s3.download_file(bucket_name,s3_key,local_file)

#use AWS S3 to upload a file in it and generate a file accessible URL
def upload_image(file_name,s3_prefix="ml-images",object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    full_object_name = f"{s3_prefix}/{object_name}"

    s3.upload_file(file_name,bucket_name,full_object_name)
    response = s3.generate_presigned_url('get_object',
                                         Params={"Bucket":bucket_name,
                                                 "Key":full_object_name},
                                                 ExpiresIn=3600)
    return response