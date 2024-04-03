import json
import os
from typing import Any

import yaml

test_trigger = True


def load_azure_config(azure_config_file: str, azure_config_secrets_file: str = None):
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
    cicd_params_file = cicd_params_file
    with open(cicd_params_file, "r") as f:
        cicd_params_dict = yaml.safe_load(f)
    return cicd_params_dict
