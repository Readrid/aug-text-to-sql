import json


def read_json(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)
