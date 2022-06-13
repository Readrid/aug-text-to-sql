import json
import sys
import model.gpt2_fn as model
from paraphrase_generation.data.utils import read_json, write_file
import re
import copy

path_to_dataset = "./../data/yelp/yelp.json"
path_to_new_dataset = "../data/yelp/aug_yelp2_v1.json"
path_to_eval_dataset = "./evaluation/eval_yelp_2v1.json"
num_of_paraphrases = 1

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 4:
        path_to_dataset = sys.argv[1]
        path_to_new_dataset = sys.argv[2]
        num_of_paraphrases = int(sys.argv[3])
    elif len(sys.argv) != 1:
        raise SyntaxError(
            "Count of arguments must 0 or 2:\n\t1. path_to_dataset\n\t2. path_to_new_dataset\n\t3. num_of_paraphrases"
        )

    data = read_json(path_to_dataset)
    result_list = []
    evaluation_list = []
    result_cnt_train_sentences = 0
    print(f"Lenght of old dataset is {len(data)}")

    for elem in data:
        result_list.append(elem)
        if elem["query-split"] != "train":
            continue

        paraphrases_dicts = []
        cnt_sentences = len(elem["sentences"])
        result_cnt_train_sentences += cnt_sentences * num_of_paraphrases
        for i_sent in range(cnt_sentences):
            sentence = elem["sentences"][i_sent]["text"]
            eval_elem_dict = {"sentence": sentence, "paraphrases": []}

            paraphrases = model.generate_paraphrase(
                model_path="./model/EleutherAI/gpt-neo-125M", sentence=sentence, num_of_paraphrases=num_of_paraphrases
            )
            variables = [name for name, value in elem["sentences"][i_sent]["variables"].items()]
            variables_dict = {}

            for paraphrase in paraphrases:
                for word in paraphrase.split(" "):
                    if any(c.isalpha() for c in word) and any(c.isdigit() for c in word):
                        if word not in variables_dict:
                            i_new_var = len(variables_dict)
                            if i_new_var >= len(variables):
                                print(f"paraphrase: {paraphrase}\nsentence: {sentence}\nparaphrases {paraphrases}")
                                continue
                            variables_dict[word] = variables[i_new_var]

            paraphrases = list(
                map(
                    lambda par: "".join(
                        variables_dict[word] if word in variables_dict else word for word in re.split(r"(\W+)", par)
                    ),
                    paraphrases,
                )
            )

            eval_elem_dict["paraphrases"] = paraphrases
            evaluation_list.append(eval_elem_dict)

            for paraphrase in paraphrases:
                new_elem = copy.deepcopy(elem["sentences"][i_sent])
                new_elem["text"] = paraphrase
                paraphrases_dicts.append(new_elem)
        elem["sentences"].extend(paraphrases_dicts)

    write_file(path_to_new_dataset, json.dumps(result_list, indent=2))
    write_file(path_to_eval_dataset, json.dumps(evaluation_list, indent=2))
    print(f"Lenght of new dataset is {len(result_list)}")
    print(f"Lenght of new dataset is {result_cnt_train_sentences}")
