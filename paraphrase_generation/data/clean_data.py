import json
import random

result_dict = {"data": []}
PARAPHRASES = "paraphrases"
SENTENCE = "sentence"
TAG = "tag"
DATA = "data"
MIN_СNT_PARAPHRASES = 1


def clean_imdb(data, file_name):
    """
    Function for cleaning imdb dataset:
    1. Discard repetitions
    2. Discard sentences with no phrases
    3. Discard phrases from one word
    4. Lead to one register
    5. Bring to a common appearance
    """

    for element in data[DATA]:
        dict = {}
        if "paraphases" not in element:
            continue
        paraphrases = list(set(element["paraphases"]))
        if len(paraphrases) < MIN_СNT_PARAPHRASES:
            continue

        sentence = element[SENTENCE].lower()
        paraphrases = list(map(lambda s: s.lower(), paraphrases))
        sentence = sentence.replace('" ', "")
        sentence = sentence.replace(' "', "")
        paraphrases = list(map(lambda s: s.replace('" ', ""), paraphrases))
        paraphrases = list(map(lambda s: s.replace(' "', ""), paraphrases))

        dict[SENTENCE] = sentence
        dict[PARAPHRASES] = paraphrases
        dict[TAG] = file_name
        result_dict["data"].append(dict)


def clean_others(data, file_name):
    """
    Function for cleaning others dataset:
    1. Discard repetitions
    2. Discard sentences with no phrases
    3. Discard phrases from one word
    4. Lead to one register
    5. Bring to a common appearance
    """

    for element in data:
        dict = {}
        element = list(set(element))
        element = list(filter(lambda s: len(s.split()) >= 2, element))

        sentence = random.choice(element)
        element.remove(sentence)

        paraphrases = element

        if len(paraphrases) < MIN_СNT_PARAPHRASES:
            continue

        sentence = sentence.lower()
        paraphrases = list(map(lambda s: s.lower(), paraphrases))

        dict[SENTENCE] = sentence
        dict[PARAPHRASES] = paraphrases
        dict[TAG] = file_name
        result_dict[DATA].append(dict)


if __name__ == "__main__":
    common_path = "./dirty/"

    file_names = ["imdb.json", "atis.json", "geography.json", "scholar.json", "spider.json"]
    size_logger = ""

    for file_name in file_names:
        with open(common_path + file_name) as jsonFile:
            data = json.load(jsonFile)
        jsonFile.close()

        old_size = len(result_dict[DATA])

        if file_name == "imdb.json":
            clean_imdb(data, file_name)
        else:
            clean_others(data, file_name)

        size_logger += f"{file_name} size is {len(result_dict[DATA]) - old_size}\n"

    size_logger += f"\tFull dataset size is {len(result_dict[DATA])}\n"
    with open("./clean/size.txt", "w") as outfile:
        outfile.write(size_logger)

    json_object = json.dumps(result_dict, indent=2)
    with open("./clean/dataset.json", "w") as outfile:
        outfile.write(json_object)
