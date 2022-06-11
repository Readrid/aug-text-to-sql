import json
import random

from paraphrase_generation.data.utils import read_json, write_file

result_dict_full = {"data": []}
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

    result_dict = {"data": []}
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

        filter_and_write_to_dict(sentence, paraphrases, dict, result_dict, result_dict_full)
    write_file("./clean/" + file_name, json.dumps(result_dict, indent=2))


def clean_others(data, file_name):
    """
    Function for cleaning others dataset:
    1. Discard repetitions
    2. Discard sentences with no phrases
    3. Discard phrases from one word
    4. Lead to one register
    5. Bring to a common appearance
    """

    result_dict = {"data": []}
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

        filter_and_write_to_dict(sentence, paraphrases, dict, result_dict, result_dict_full)

    write_file("./clean/" + file_name, json.dumps(result_dict, indent=2))


def filter_and_write_to_dict(sentence, paraphrases, dict, result_dict, result_dict_full):
    paraphrases = list(set(paraphrases))
    if sentence in paraphrases:
        paraphrases.remove(sentence)

    dict[SENTENCE] = sentence
    dict[PARAPHRASES] = paraphrases
    result_dict[DATA].append(dict)
    dict_full = dict.copy()
    dict_full[TAG] = file_name
    result_dict_full["data"].append(dict_full)


if __name__ == "__main__":
    common_path = "./dirty/"

    file_names = ["imdb.json", "atis.json", "geography.json", "scholar.json", "spider.json"]
    size_logger = ""

    for file_name in file_names:
        data = read_json(common_path + file_name)

        old_size = len(result_dict_full[DATA])

        if file_name == "imdb.json":
            clean_imdb(data, file_name)
        else:
            clean_others(data, file_name)

        size_logger += f"{file_name} size is {len(result_dict_full[DATA]) - old_size}\n"

    size_logger += f"\tFull dataset size is {len(result_dict_full[DATA])}\n"
    write_file("./clean/size.txt", size_logger)

    write_file("./clean/dataset.json", json.dumps(result_dict_full, indent=2))
