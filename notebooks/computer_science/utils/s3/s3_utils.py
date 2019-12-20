import io
import datetime
import json
import pickle

import boto3
import pandas as pd
import xml.etree.ElementTree as ET

from notebooks.computer_science.utils.general_utils import to_str

INTUITIVE_ML_BUCKET = 'intuitiveml-data-sets'

s3_client = boto3.client('s3')


def key_prefix(key):
    """Returns the key prefix that can be used to find the latest key with that given prefix."""
    return key.split('{')[0]


def s3_ls(globlike, order_by=None):
    """Returns a list of keys"""
    
    bucket, keyglob = globlike.split('/', 1)
    prefix = keyglob.split('*', 1)[0]
    response = boto3.client('s3').list_objects(Bucket=bucket, Prefix=prefix)
    if response.get('Contents'):
        if order_by and order_by in ('Key', 'LastModified', 'Size'):
            results = sorted(response['Contents'], key=lambda obj: obj[order_by])
        else:
            results = response['Contents']
        return [obj['Key'] for obj in results]
    return []


def latest_key(bucket_name, key_prefix, version=1):
    """Returns most recent key."""
    
    key_map = {1: -1, 2: -2}
    
    latest_key = s3_ls(f"{bucket_name}/{key_prefix}*", order_by="Key")[key_map[version]]
    return latest_key


def latest_key_full_path(bucket, key_prefix, version=1):
    """Returns the most recent key's full path to s3."""
    return f's3://{INTUITIVE_ML_BUCKET}/{latest_key(bucket, key_prefix)}'


def upload_versioned_df_to_s3(df, key, bucket=INTUITIVE_ML_BUCKET):
    """
    Uploads a versioned (with date) df to s3 as a csv. Key must contain `{date}`.
    """
    RUN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    key = key.format(date=RUN_DATE)
    
    return s3_client.put_object(
        Body=df.to_csv(index=False), 
        Bucket=bucket,
        Key=key,
    )


def upload_versioned_pickle_to_s3(obj, key, bucket=INTUITIVE_ML_BUCKET):
    """
    Uploads a versioned (with date) object to s3 after pickling (as bytes). Key must contain `{date}`.
    """
    RUN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    key = key.format(date=RUN_DATE)
    
    pickle_byte_obj = pickle.dumps(obj)
    
    return s3_client.put_object(
        Body=pickle_byte_obj, 
        Bucket=bucket,
        Key=key,
    )


def get_all_s3_objects(s3, **base_kwargs):
    """
    Gathers all object keys inside of S3 bucket.
    """
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  
            break
        continuation_token = response.get('NextContinuationToken')
        
        
def get_obj_s3(key=''):
    """
    Get object from S3, read contents as bytes, convert to string representation.
    """
    obj = s3_client.get_object(Bucket=INTUITIVE_ML_BUCKET, Key=key)
    obj_bytes = obj['Body'].read()
    obj_str = to_str(obj_bytes)
    return obj_str


def load_pickle_from_s3(key=''):
    """
    Get pickled object from S3, read contents as bytes, load via pickle.
    """
    obj = s3_client.get_object(Bucket=INTUITIVE_ML_BUCKET, Key=key)
    obj_bytes = obj['Body'].read()
    
    return pickle.loads(obj_bytes)


def load_csv_from_s3(bucket=INTUITIVE_ML_BUCKET, key=''):
    """
    Get object from S3, read contents as bytes, convert to string representation.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    obj_bytes = obj['Body'].read()
    return pd.read_csv(io.BytesIO(obj_bytes), dtype=str)


def load_json_from_s3(key=''):
    """
    Get object from s3, load as JSON.
    """
    obj_str = get_obj_s3(key=key)
    return json.loads(obj_str)
    

def get_xml_survey_from_s3(key=''):
    """
    Get xml survey from S3 and return as ET root. 
    """
    obj_str = get_obj_s3(key=key)
    root = ET.fromstring(obj_str)
    return root
