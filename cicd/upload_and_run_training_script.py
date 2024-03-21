import argparse
import os

import yaml
from azure.ai.ml import Input, MLClient, command
from azure.identity import EnvironmentCredential
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--config-secrets-file", type=str)
    args = parser.parse_args()
    config_file = args.config_file
    config_secrets_file = args.config_secrets_file

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    azure_config_dict = config_dict["azure_ml"]["dev"]
    upload_training_script_params = config_dict["upload_training_script_params"]
    print(azure_config_dict)

    os.environ["AZURE_TENANT_ID"] = azure_config_dict["service_principal_tenant_id"]
    os.environ["AZURE_CLIENT_ID"] = azure_config_dict["service_principal_client_id"]
    os.environ["AZURE_ML_SUBSCRIPTION_ID"] = azure_config_dict["azure_subscription_id"]
    os.environ["DEV_AZURE_ML_WORKSPACE_NAME"] = azure_config_dict["workspace_name"]
    os.environ["DEV_AZURE_ML_RESOURCE_GROUP_NAME"] = azure_config_dict[
        "resource_group_name"
    ]

    if config_secrets_file is not None:
        # in the pipeline the secrets will be saved
        # if run locally, argument azure-config-secrets-file must be provided
        with open(config_secrets_file, "r") as f:
            secrets_config_dict = yaml.safe_load(f)
        azure_secrets_config_dict = secrets_config_dict["azure_ml"]["dev"]
        os.environ["AZURE_CLIENT_SECRET"] = azure_secrets_config_dict[
            "service_principal_client_secret"
        ]
        print(azure_secrets_config_dict)

    sp_auth = ServicePrincipalAuthentication(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        service_principal_id=os.environ["AZURE_CLIENT_ID"],
        service_principal_password=os.environ["AZURE_CLIENT_SECRET"],
    )
    ws = Workspace(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group=os.environ["DEV_AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["DEV_AZURE_ML_WORKSPACE_NAME"],
        auth=sp_auth,
    )

    ml_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["DEV_AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["DEV_AZURE_ML_WORKSPACE_NAME"],
        credential=EnvironmentCredential(),
    )

    print(upload_training_script_params)

    os.environ["TRAINING_ENVIRONMENT_NAME"] = upload_training_script_params[
        "training_environment_name"
    ]

    env_version = Environment.list(ws)[os.environ["TRAINING_ENVIRONMENT_NAME"]].version
    print(f"using environment {os.environ['TRAINING_ENVIRONMENT_NAME']}:{env_version}")

    num_epochs = upload_training_script_params["num_epochs"]
    batch_size = upload_training_script_params["batch_size"]

    print(f"num epochs: {num_epochs}, batch_size: {batch_size}")

    command_job = command(
        code=os.path.join("mtg_artist_classifier", "classifier"),
        command=f"python train_model.py --train-data-folder ${{inputs.train_data_folder}} --val-data-folder ${{inputs.val_data_folder}} --batch-size {batch_size} --num-epochs {num_epochs}",
        environment=f"{os.environ['TRAINING_ENVIRONMENT_NAME']}:{env_version}",
        inputs={
            "train_data_folder": Input(
                type="uri_folder",
                path=upload_training_script_params["training_data_folder"],
            ),
            "val_data_folder": Input(
                type="uri_folder",
                path=upload_training_script_params["validation_data_folder"],
            ),
        },
        compute=upload_training_script_params["training_compute_name"],
        experiment_name=upload_training_script_params["experiment_name"],
    )

    returned_job = ml_client.jobs.create_or_update(command_job)
    ml_client.jobs.stream(returned_job.name)
