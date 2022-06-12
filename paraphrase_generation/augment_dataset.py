import json
import sys
import model.gpt2_fn as model
from data_processing.utils import subtle_query
from paraphrase_generation.data.utils import read_json, write_file
import re
import copy

path_to_dataset = "./../data/yelp/yelp.json"
path_to_new_dataset = "../data/yelp/aug_yelp2.json"
path_to_eval_dataset = "./evaluation/eval_yelp.json"
num_of_paraphrases = 1

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 4:
        path_to_dataset = sys.argv[1]
        path_to_new_dataset = sys.argv[2]
        num_of_paraphrases = int(sys.argv[3])
    elif len(sys.argv) != 1:
        raise SyntaxError("Count of arguments must 0 or 2:\n\t1. path_to_dataset\n\t2. path_to_new_dataset\n\t3. num_of_paraphrases")

    data = read_json(path_to_dataset)
    result_list = []
    evaluation_list = []
    print(f"Lenght of old dataset is {len(data)}")

    for elem in data:
        result_list.append(elem)
        curr_elem_dict = {"sentence": elem["sentences"][0]["text"], "paraphrases": []}
        if len(elem["sentences"]) > 1:
            print(elem["sentences"])

        sentence = elem["sentences"][0]["text"]
        # sentence = subtle_query(sentence, elem["sentences"][0]["variables"])

        paraphrases = model.generate_paraphrase(model_path="./model/EleutherAI/gpt-neo-125M",
                                                sentence=sentence,
                                                num_of_paraphrases=num_of_paraphrases)
        variables = [x["name"] for x in elem["variables"]]
        variables_dict = {}

        for paraphrase in paraphrases:
            for word in paraphrase.split(" "):
                if any(c.isalpha() for c in word) and any(c.isdigit() for c in word):
                    if word not in variables_dict:
                        i = len(variables_dict)
                        if i >= len(variables):
                            print(f"paraphrase: {paraphrase}\nelem: {elem['sentences'][0]['text']}\nparaphrases {paraphrases}")
                            continue
                        variables_dict[word] = variables[i]

        paraphrases = list(map(lambda par: ''.join(
            variables_dict[word] if word in variables_dict else word for word in re.split(r'(\W+)', par)), paraphrases))

        curr_elem_dict["paraphrases"] = paraphrases
        evaluation_list.append(curr_elem_dict)
        for paraphrase in paraphrases:
            new_elem = copy.deepcopy(elem)
            new_elem["sentences"][0]["text"] = paraphrase
            result_list.append(new_elem)


    write_file(path_to_new_dataset, json.dumps(result_list, indent=2))
    write_file(path_to_eval_dataset, json.dumps(evaluation_list, indent=2))
    print(f"Lenght of new dataset is {len(result_list)}")
