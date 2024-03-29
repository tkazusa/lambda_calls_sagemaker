# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains code to create and manage SageMaker ``MultiDataModel``"""
from __future__ import absolute_import

import os
from six.moves.urllib.parse import urlparse

import sagemaker
from sagemaker import s3
from sagemaker.model import Model
from sagemaker.session import Session

MULTI_MODEL_CONTAINER_MODE = "MultiModel"


class MultiDataModel(Model):
    """A SageMaker ``MultiDataModel`` that can be used to deploy multiple models to the same
    SageMaker ``Endpoint``, and also deploy additional models to an existing SageMaker
    multi-model ``Endpoint``
    """

    def __init__(
        self,
        name,
        model_data_prefix,
        model=None,
        image=None,
        role=None,
        sagemaker_session=None,
        **kwargs
    ):
        """Initialize a ``MultiDataModel``. In addition to these arguments, it supports all
           arguments supported by ``Model`` constructor

        Args:
            name (str): The model name.
            model_data_prefix (str): The S3 prefix where all the models artifacts (.tar.gz)
                in a Multi-Model endpoint are located
            model (sagemaker.Model): The Model object that would define the
                SageMaker model attributes like vpc_config, predictors, etc.
                If this is present, the attributes from this model are used when
                deploying the ``MultiDataModel``.  Parameters 'image', 'role' and 'kwargs'
                are not permitted when model parameter is set.
            image (str): A Docker image URI. It can be null if the 'model' parameter
                is passed to during ``MultiDataModel`` initialization (default: None)
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field or if the 'model' parameter
                is passed to during ``MultiDataModel`` initialization (default: None)
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            **kwargs: Keyword arguments passed to the
                :class:`~sagemaker.model.Model` initializer.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.Model`.
        """
        # Validate path
        if not model_data_prefix.startswith("s3://"):
            raise ValueError(
                'Expecting S3 model prefix beginning with "s3://". Received: "{}"'.format(
                    model_data_prefix
                )
            )

        if model and (image or role or kwargs):
            raise ValueError(
                "Parameters image, role or kwargs are not permitted when model parameter is passed."
            )

        self.name = name
        self.model_data_prefix = model_data_prefix
        self.model = model
        self.container_mode = MULTI_MODEL_CONTAINER_MODE
        self.sagemaker_session = sagemaker_session or Session()
        self.s3_client = self.sagemaker_session.boto_session.client("s3")

        # Set the ``Model`` parameters if the model parameter is not specified
        if not self.model:
            super(MultiDataModel, self).__init__(
                self.model_data_prefix,
                image,
                role,
                name=self.name,
                sagemaker_session=self.sagemaker_session,
                **kwargs
            )

    def prepare_container_def(self, instance_type, accelerator_type=None):
        """Return a container definition set with MultiModel mode,
        model data and other parameters from the model (if available).

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Returns:
            dict[str, str]: A complete container definition object usable with the CreateModel API
        """
        # Copy the trained model's image and environment variables if they exist. Models trained
        # with FrameworkEstimator set framework specific environment variables which need to be
        # copied over
        if self.model:
            container_definition = self.model.prepare_container_def(instance_type, accelerator_type)
            image = container_definition["Image"]
            environment = container_definition["Environment"]
        else:
            image = self.image
            environment = self.env
        return sagemaker.container_def(
            image,
            env=environment,
            model_data_url=self.model_data_prefix,
            container_mode=self.container_mode,
        )

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        accelerator_type=None,
        endpoint_name=None,
        update_endpoint=False,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
    ):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
        ``Endpoint`` from this ``Model``. If self.model is not None, then the ``Endpoint``
        will be deployed with parameters in self.model (like vpc_config,
        enable_network_isolation, etc).  If self.model is None, then use the parameters
        in ``MultiDataModel`` constructor will be used. If ``self.predictor_cls`` is not
        None, this method returns a the result of invoking ``self.predictor_cls`` on
        the created endpoint name.

        The name of the created model is accessible in the ``name`` field of
        this ``Model`` after deploy returns

        The name of the created endpoint is accessible in the
        ``endpoint_name`` field of this ``Model`` after deploy returns.

        Args:
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``.
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode.
            accelerator_type (str): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            update_endpoint (bool): Flag to update the model in an existing
                Amazon SageMaker endpoint. If True, this will deploy a new
                EndpointConfig to an already existing endpoint and delete
                resources corresponding to the previous EndpointConfig. If
                False, a new endpoint will be created. Default: False
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            wait (bool): Whether the call should wait until the deployment of
                this model completes (default: True).
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
                ``self.predictor_cls`` on the created endpoint name,
                if ``self.predictor_cls``
                is not None. Otherwise, return None.
        """
        # Set model specific parameters
        if self.model:
            enable_network_isolation = self.model.enable_network_isolation()
            role = self.model.role
            vpc_config = self.model.vpc_config
            predictor = self.model.predictor_cls
        else:
            enable_network_isolation = self.enable_network_isolation()
            role = self.role
            vpc_config = self.vpc_config
            predictor = self.predictor_cls

        if role is None:
            raise ValueError("Role can not be null for deploying a model")

        container_def = self.prepare_container_def(instance_type, accelerator_type=accelerator_type)
        self.sagemaker_session.create_model(
            self.name,
            role,
            container_def,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            tags=tags,
        )

        production_variant = sagemaker.production_variant(
            self.name, instance_type, initial_instance_count, accelerator_type=accelerator_type
        )
        if endpoint_name:
            self.endpoint_name = endpoint_name
        else:
            self.endpoint_name = self.name

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        if update_endpoint:
            endpoint_config_name = self.sagemaker_session.create_endpoint_config(
                name=self.name,
                model_name=self.name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                tags=tags,
                kms_key=kms_key,
                data_capture_config_dict=data_capture_config_dict,
            )
            self.sagemaker_session.update_endpoint(self.endpoint_name, endpoint_config_name)
        else:
            self.sagemaker_session.endpoint_from_production_variants(
                name=self.endpoint_name,
                production_variants=[production_variant],
                tags=tags,
                kms_key=kms_key,
                wait=wait,
                data_capture_config_dict=data_capture_config_dict,
            )

        if predictor:
            return predictor(self.endpoint_name, self.sagemaker_session)
        return None

    def add_model(self, model_data_source, model_data_path=None):
        """Adds a model to the ``MultiDataModel`` by uploading or copying the model_data_source
         artifact to the given S3 path model_data_path relative to model_data_prefix

        Args:
            model_source: Valid local file path or S3 path of the trained model artifact
            model_data_path: S3 path where the trained model artifact
                should be uploaded relative to ``self.model_data_prefix`` path. (default: None).
                If None, then the model artifact is uploaded to a path relative to model_data_prefix

        Returns:
            str: S3 uri to uploaded model artifact
        """
        parse_result = urlparse(model_data_source)

        # If the model source is an S3 path, copy the model artifact to the destination S3 path
        if parse_result.scheme == "s3":
            source_bucket, source_model_data_path = s3.parse_s3_url(model_data_source)
            copy_source = {"Bucket": source_bucket, "Key": source_model_data_path}

            if not model_data_path:
                model_data_path = source_model_data_path

            # Construct the destination path
            dst_url = os.path.join(self.model_data_prefix, model_data_path)
            destination_bucket, destination_model_data_path = s3.parse_s3_url(dst_url)

            # Copy the model artifact
            self.s3_client.copy(copy_source, destination_bucket, destination_model_data_path)
            return os.path.join("s3://", destination_bucket, destination_model_data_path)

        # If the model source is a local path, upload the local model artifact to the destination
        #  s3 path
        if os.path.exists(model_data_source):
            destination_bucket, dst_prefix = s3.parse_s3_url(self.model_data_prefix)
            if model_data_path:
                dst_s3_uri = os.path.join(dst_prefix, model_data_path)
            else:
                dst_s3_uri = os.path.join(dst_prefix, os.path.basename(model_data_source))
            self.s3_client.upload_file(model_data_source, destination_bucket, dst_s3_uri)
            # return upload_path
            return os.path.join("s3://", destination_bucket, dst_s3_uri)

        # Raise error if the model source is of an unexpected type
        raise ValueError(
            "model_source must either be a valid local file path or s3 uri. Received: "
            '"{}"'.format(model_data_source)
        )

    def list_models(self):
        """Generates and returns relative paths to model archives stored at model_data_prefix
        S3 location.

        Yields: Paths to model archives relative to model_data_prefix path.
        """
        bucket, url_prefix = s3.parse_s3_url(self.model_data_prefix)
        file_keys = self.sagemaker_session.list_s3_files(bucket=bucket, key_prefix=url_prefix)
        for file_key in file_keys:
            # Return the model paths relative to the model_data_prefix
            # Ex: "a/b/c.tar.gz" -> "b/c.tar.gz" where url_prefix = "a/"
            yield file_key.replace(url_prefix, "")
