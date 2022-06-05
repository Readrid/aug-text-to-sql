from typing import Dict

from tqdm import tqdm


def subtle_query(text: str, variables: Dict[str, str]) -> str:
    for variable in variables.items():
        text = text.replace(variable[0], variable[1])
    return text


def subtle_sql(text: str, variables):
    if type(variables) is list:
        return list(map(lambda variable: subtle_query(text, variable), variables))
    else:
        return subtle_query(text, variables)


def make_canonical(query):
    return query


def preprocess_data(data):
    not_annotated_data = {
        "train": {"sentences": [], "sql": []},
        "dev": {"sentences": [], "sql": []},
        "test": {"sentences": [], "sql": []},
    }
    annotated_data = {
        "train": {"sentences": [], "sql": []},
        "dev": {"sentences": [], "sql": []},
        "test": {"sentences": [], "sql": []},
    }

    for elem in data:
        sentences = [subtle_query(obj["text"], obj["variables"]) for obj in elem["sentences"]]
        sql = subtle_sql(elem["sql"][0], [obj["variables"] for obj in elem["sentences"]])
        not_annotated_data[elem["query-split"]]["sentences"].extend(sentences)
        not_annotated_data[elem["query-split"]]["sql"].extend(sql)

    for elem in tqdm(data):
        variables = [obj["variables"] for obj in elem["sentences"]]
        sentences = [subtle_query(obj["text"], obj["variables"]) for obj in elem["sentences"]]
        sql = [elem["sql"][0] for _ in elem["sentences"]]

        sql_processed = []
        for (variable_set, sql_query) in zip(variables, sql):
            sql_processed.append(make_canonical(sql_query))  # do canonicalization

        sql_processed = [subtle_sql(sql_processed[i], variables[i]) for i in range(len(elem["sentences"]))]

        annotated_data[elem["query-split"]]["sentences"].extend(sentences)
        annotated_data[elem["query-split"]]["sql"].extend(sql_processed)

    return not_annotated_data, annotated_data
