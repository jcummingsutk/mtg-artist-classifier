import argparse
import json
import os

import yaml
from azure.ai.ml import Input, MLClient, command
from azure.identity import EnvironmentCredential
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--azure-config-file", type=str)
    parser.add_argument("--azure-config-secrets-file", type=str)
    parser.add_argument("--cicd-params-file", type=str)
    args = parser.parse_args()
    azure_config_file = args.azure_config_file
    azure_config_secrets_file = args.azure_config_secrets_file
    cicd_params_file = args.cicd_params_file

    with open(azure_config_file, "r") as f:
        azure_config_dict = json.load(fp=f)

    os.environ["AZURE_TENANT_ID"] = azure_config_dict["SERVICE_PRINCIPAL_TENANT_ID"]
    os.environ["AZURE_CLIENT_ID"] = azure_config_dict["SERVICE_PRINCIPAL_CLIENT_ID"]
    os.environ["AZURE_ML_SUBSCRIPTION_ID"] = azure_config_dict["AZURE_SUBSCRIPTION_ID"]
    os.environ["DEV_AZURE_ML_WORKSPACE_NAME"] = azure_config_dict["DEV_WORKSPACE_NAME"]
    os.environ["DEV_AZURE_ML_RESOURCE_GROUP_NAME"] = azure_config_dict[
        "RESOURCE_GROUP_NAME"
    ]

    if azure_config_secrets_file is not None:
        # in the pipeline the secrets will be saved
        # if run locally, argument azure-config-secrets-file must be provided
        with open(azure_config_secrets_file, "r") as f:
            azure_secrets_config_dict = json.load(fp=f)
        os.environ["AZURE_CLIENT_SECRET"] = azure_secrets_config_dict[
            "SERVICE_PRINCIPAL_CLIENT_SECRET"
        ]

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

    with open(cicd_params_file, "r") as fp:
        cicd_params_dict = yaml.safe_load(fp)
    upload_training_script_params = cicd_params_dict["upload_training_script_params"]
    print(upload_training_script_params)

    os.environ["TRAINING_ENVIRONMENT_NAME"] = upload_training_script_params[
        "training_environment_name"
    ]

    env_version = Environment.list(ws)[os.environ["TRAINING_ENVIRONMENT_NAME"]].version
    print(f"using environment {os.environ['TRAINING_ENVIRONMENT_NAME']}:{env_version}")

    command_job = command(
        code=os.path.join("mtg_artist_classifier", "classifier"),
        command="python train_model.py --train-data-folder ${{inputs.train_data_folder}} --val-data-folder ${{inputs.val_data_folder}} --batch-size 4 --num-epochs 25",
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
