import json


def read_json(path: str):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data

def write_file(path: str, data):
    with open(path, "w") as outfile:
        outfile.write(data)
    outfile.close()
