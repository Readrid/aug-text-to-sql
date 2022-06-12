import json
from typing import Dict, List, Tuple

# TODO: merge with evaluation


def add_semicolon(query: str) -> str:
    """
    Adds a semicolon at the end of the SQL statement if it is missing.
    :param query:
    :return: processed query
    """

    query = query.strip()
    if len(query) > 0 and query[-1] != ";":
        return query + ";"
    return query


def standardise_blank_spaces(query: str) -> str:
    """
    Ensures there is one blank space between each special character and word.
    :param query:
    :return: processed query
    """

    return " ".join(query.split())


def capitalise(query: str, variables: List[str] = list()) -> str:
    """
    Converts all non-quoted sections of the query to uppercase.
    :param query:
    :param variables: query variables
    :return: processed query
    """

    def update_quotes(char, in_single: bool, in_double: bool) -> Tuple[bool, bool]:
        if char == '"' and not in_single:
            in_double = not in_double
        elif char == "'" and not in_double:
            in_single = not in_single
        return in_single, in_double

    new_tokens = []
    in_single_quote, in_double_quote = False, False
    for token in query.split():
        if token in variables:
            new_tokens.append(token)
        else:
            new_token = []
            for char in token:
                if in_single_quote or in_double_quote:
                    new_token.append(char)
                else:
                    new_token.append(char.lower())

                in_single_quote, in_double_quote = update_quotes(char, in_single_quote, in_double_quote)
            new_tokens.append("".join(new_token))

    return " ".join(new_tokens)


def make_canonical(query: str, variables: List[str]) -> str:
    query = add_semicolon(query)
    query = standardise_blank_spaces(query)
    query = capitalise(query, variables)
    return query


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

    for (key, variables), (_, queries) in zip(variables.items(), queries.items()):
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


def concat(info):
    return " ".join(info)


def type2canon(type_name):
    return type_name.split("(")[0]


def load_json(file_name: str):
    with open(file_name, "r") as f:
        dirty_json_data = json.load(f)
    return dirty_json_data
