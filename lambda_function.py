# -*- coding: utf-8 -*-
import json
from datetime import datetime

import boto3
from sagemaker.tensorflow import TensorFlow

dataset_location = 's3://sagemaker-us-east-1-815969174475/data/DEMO-cifar10'
role = 'arn:aws:iam::815969174475:role/service-role/AmazonSageMaker-ExecutionRole-20191202T145588'


def lambda_handler(event, context):
    s3 = boto3.resource('s3')  # S3オブジェクトを取得
    bucket = s3.Bucket('sagemaker-us-east-1-815969174475')
    bucket.download_file('cifar10_keras_sm_sample0.py',
                         'cifar10_keras_sm_sample0.py')
    now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    training_job_name = 'training-tf-test-{}'.format(now_str)

    estimator = TensorFlow(base_job_name=training_job_name,
                           entry_point='cifar10_keras_sm_sample0.py',
                           role=role,
                           framework_version='1.12.0',
                           py_version='py3',
                           hyperparameters={'epochs': 5},
                           train_instance_count=1,
                           train_instance_type='ml.p2.xlarge')

    estimator.fit({'train': '{}/train'.format(dataset_location),
                   'validation': '{}/validation'.format(dataset_location),
                   'eval': '{}/eval'.format(dataset_location)})

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


res = lambda_handler(None, None)
