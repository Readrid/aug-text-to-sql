from typing import Dict

from tqdm import tqdm

from evaluation.canonicalization import make_canonical


def subtle_query(text: str, variables: Dict[str, str]) -> str:
    for variable in variables.items():
        text = text.replace(variable[0], variable[1])
    return text


def subtle_sql(text: str, variables):
    if type(variables) is list:
        return list(map(lambda variable: subtle_query(text, variable), variables))
    else:
        return subtle_query(text, variables)


def preprocess_data(data):
    processed_data = {
        "train": {"sentences": [], "sql": [], "variables": []},
        "dev": {"sentences": [], "sql": [], "variables": []},
        "test": {"sentences": [], "sql": [], "variables": []},
    }

    for elem in data:
        variables = [obj["variables"] for obj in elem["sentences"]]
        sentences = [subtle_query(obj["text"], obj["variables"]) for obj in elem["sentences"]]
        sql = [elem["sql"][0] for _ in elem["sentences"]]

        processed_data[elem["query-split"]]["sentences"].extend(sentences)
        processed_data[elem["query-split"]]["sql"].extend(sql)
        processed_data[elem["query-split"]]["variables"].extend(variables)

    return processed_data


def get_sql(data: Dict):
    sql = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for (key, value) in data.items():
        for (variable_set, sql_query) in zip(value["variables"], value["sql"]):
            sql[key].append(subtle_sql(sql_query, variable_set))

    return sql


def get_sql_substitution(data: Dict):
    sql = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for (key, value) in data.items():
        for (variable_set, sql_query) in zip(value["variables"], value["sql"]):
            sql[key].append(subtle_sql(make_canonical(sql_query, variable_set), variable_set))

    return sql


def substitute_variables(variables: Dict, queries: Dict):
    sql = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for (key, variables, _, queries) in zip(variables.items(), queries.items()):
        for (variable_set, sql_query) in zip(variables, queries):
            sql[key].append(subtle_sql(make_canonical(sql_query, variable_set), variable_set))

    return sql


def get_variables(data: Dict):
    variables = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for (key, value) in data.items():
        variables[key] = value["variables"]

    return variables
