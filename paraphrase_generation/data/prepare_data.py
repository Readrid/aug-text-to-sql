import json

if __name__ == "__main__":
    common_path = "./dirty/"
    file_name = "imdb.json"
    with open(common_path + file_name) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()

    i = 0
    result_dict = {"data": []}
    for sentences in data:
        dict = {}
        for i, text in enumerate(sentences["sentences"]):
            if i == 0:
                dict["sentence"] = text["text"]
            elif i == 1:
                dict["paraphases"] = [text["text"]]

            else:
                dict["paraphases"].append(text["text"])
        result_dict["data"].append(dict)
        dict = {}

    json_object = json.dumps(result_dict, indent=2)
    with open(file_name, "w") as outfile:
        outfile.write(json_object)
