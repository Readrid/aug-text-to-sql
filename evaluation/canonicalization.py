from typing import Tuple, List


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
