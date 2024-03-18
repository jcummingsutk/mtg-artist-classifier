import argparse


def get_requirmements_as_list(requirements_file: str) -> list[str]:
    with open(requirements_file, "r") as f:
        lines = f.readlines()
        print(lines)
    return lines


def separate_torch_requirements(
    requirements_as_list: list[str],
) -> tuple[list[str], list[str]]:
    non_torch_requirements = []
    torch_requirements = []
    for requirement in requirements_as_list:
        if "torch" in requirement:
            torch_requirements.append(requirement)
        else:
            non_torch_requirements.append(requirement)
    return non_torch_requirements, torch_requirements


def write_requirements_file(requirements_as_list: list[str], output_file: str):
    with open(output_file, "w") as output_file:
        for line in requirements_as_list:
            output_file.write(line)


def modify_requirements_file(
    requirements_file: str,
    torch_output_requirements_filename: str,
    non_torch_output_requirements_filename: str,
):
    requirements_as_list = get_requirmements_as_list(requirements_file)

    non_torch_requirements, torch_requirements = separate_torch_requirements(
        requirements_as_list
    )
    write_requirements_file(torch_requirements, torch_output_requirements_filename)
    write_requirements_file(
        non_torch_requirements, non_torch_output_requirements_filename
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements-file", type=str)
    parser.add_argument("--torch-output-requirements-filename", type=str)
    parser.add_argument("--non-torch-output-requirements-filename")
    args = parser.parse_args()
    requirements_file = args.requirements_file
    torch_output_requirements_filename = args.torch_output_requirements_filename
    non_torch_output_requirements_filename = args.non_torch_output_requirements_filename
