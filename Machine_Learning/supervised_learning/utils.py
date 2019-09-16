import numpy as np
import pandas
import boto3
import datetime


s3 = boto3.client('s3')


def get_csv_from_s3(key, bucket="intuitiveml-data-sets"):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pandas.read_csv(obj["Body"])


def get_obj_s3(key, bucket="intuitiveml-data-sets"):
    obj = s3.get_object(Bucket=bucket, Key=key)["Body"]
    return obj


def get_mnist_data(limit=None):
    print("Reading and Transforming MNIST Data...")
    df = get_csv_from_s3("mnist_train.csv")
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y
