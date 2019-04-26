import pandas
import boto3


s3 = boto3.client('s3')


def get_csv_from_s3(key, bucket="intuitiveml-data-sets"):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pandas.read_csv(obj["Body"])


def get_obj_s3(key, bucket="intuitiveml-data-sets"):
    obj = s3.get_object(Bucket=bucket, Key=key)["Body"]
    return obj