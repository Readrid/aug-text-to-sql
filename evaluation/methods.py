from psycopg2 import OperationalError


def execute_query(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        return cursor.execute(query)
    except OperationalError as _:
        return None


def execution_accuracy(connection, pred_queries, real_queries):
    number_matches = 0

    for (pred_query, real_query) in zip(pred_queries, real_queries):
        pred_result = execute_query(connection, pred_query)
        real_result = execute_query(connection, real_query)
        if pred_result == real_result and pred_result is not None:
            number_matches += 1

    return number_matches / len(pred_queries)


def logical_form_accuracy(pred_queries, real_queries):
    return sum(map(lambda pred_query, real_query: pred_query == real_query, zip(pred_queries, real_queries))) / len(
        pred_queries
    )
