import argparse
import json
import os
import shutil
from typing import Any

import yaml
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential


def save_azure_config(azure_config_file: str, azure_config_secrets_file: str = None):
    with open(azure_config_file, "r") as f:
        azure_config_dict = json.load(fp=f)

    os.environ["AZURE_TENANT_ID"] = azure_config_dict["SERVICE_PRINCIPAL_TENANT_ID"]
    os.environ["AZURE_CLIENT_ID"] = azure_config_dict["SERVICE_PRINCIPAL_CLIENT_ID"]
    os.environ["AZURE_SUBSCRIPTION_ID"] = azure_config_dict["AZURE_SUBSCRIPTION_ID"]
    os.environ["RESOURCE_GROUP_NAME"] = azure_config_dict["RESOURCE_GROUP_NAME"]
    os.environ["AZURE_DEV_WORKSPACE_NAME"] = azure_config_dict["DEV_WORKSPACE_NAME"]
    if azure_config_secrets_file is not None:
        with open(azure_config_secrets_file, "r") as f:
            azure_secrets_config_dict = json.load(fp=f)
        os.environ["AZURE_CLIENT_SECRET"] = azure_secrets_config_dict[
            "SERVICE_PRINCIPAL_CLIENT_SECRET"
        ]


def get_cicd_config(cicd_params_file: str) -> dict[str, Any]:
    cicd_params_file = args.cicd_params_file
    with open(cicd_params_file, "r") as f:
        cicd_params_dict = yaml.safe_load(f)
    print(cicd_params_dict)
    return cicd_params_dict


def download_model(
    ml_client: MLClient,
    model_name_to_download: str,
    model_version_to_download: str,
    model_download_location: str,
):
    ml_client.models.download(
        name=model_name_to_download,
        version=model_version_to_download,
        download_path=model_download_location,
    )


def copy_model_folder(source: str, dest: str):
    shutil.copytree(source, dest, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cicd-params-file", type=str)
    parser.add_argument("--azure-config-file", type=str)
    parser.add_argument("--azure-config-secrets-file", type=str)
    parser.add_argument("--model-name-to-download", type=str)
    parser.add_argument("--model-version-to-download", type=str)
    parser.add_argument("--model-copy-destination")
    args = parser.parse_args()

    save_azure_config(args.azure_config_file, args.azure_config_secrets_file)
    cicd_params_dict = get_cicd_config(args.cicd_params_file)

    download_params = cicd_params_dict["download_params"]
    model_download_location = download_params["model_download_location"]

    environment_credential = EnvironmentCredential()

    ml_client = MLClient(
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["RESOURCE_GROUP_NAME"],
        credential=environment_credential,
        workspace_name=os.environ["AZURE_DEV_WORKSPACE_NAME"],
    )

    download_model(
        ml_client,
        args.model_name_to_download,
        args.model_version_to_download,
        model_download_location,
    )

    copy_model_folder(
        os.path.join(model_download_location, args.model_name_to_download),
        args.model_copy_destination,
    )
