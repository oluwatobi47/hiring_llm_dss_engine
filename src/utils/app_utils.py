import json
from uuid import uuid4

from fastapi import FastAPI


def generate_unique_id():
    """
    Generate a unique ID and return it as a string.

    Returns:
        str: A string with a unique ID.
    """
    return str(uuid4())


def read_json(json_file_path: str) -> dict:
    with open(json_file_path) as json_file:
        return json.load(json_file)


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

