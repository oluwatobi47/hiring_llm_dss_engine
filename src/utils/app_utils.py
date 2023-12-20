import json
from typing import Union
from uuid import uuid4

from fastapi import FastAPI


def generate_unique_id():
    """
    Generate a unique ID and return it as a string.

    Returns:
        str: A string with a unique ID.
    """
    return str(uuid4())


def read_json(json_file_path: str) -> Union[dict, list]:
    with open(json_file_path) as json_file:
        return json.load(json_file)


def save_json(json_data: dict, file_path_and_name: str):
    with open(f"{file_path_and_name}", 'w') as f:
        json.dump(json_data, f, indent=2)
        f.close()


def pair_list_items(list1: list, list2: list) -> list:
    output = []
    for item in list1:
        for item2 in list2:
            output.append([item, item2])
    return output


def load_app_routes(app: FastAPI, routes: list):
    for route in routes:
        app.include_router(
            route['router'],
            prefix=route['prefix'],
            tags=route['tags']
        )


def get_file_name(link, alternate=None):
    last_slash_index = link.rfind("/")  # Find the last slash from the right

    if last_slash_index != -1:
        return link[last_slash_index + 1:]  # Extract characters after the slash
    else:
        return link if alternate is None else alternate  # Return the entire string if no slash is found
