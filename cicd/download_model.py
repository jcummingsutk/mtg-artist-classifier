import argparse
import os
import shutil

from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential
from utilities import get_cicd_config, load_azure_config


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

    load_azure_config(args.azure_config_file, args.azure_config_secrets_file)
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
