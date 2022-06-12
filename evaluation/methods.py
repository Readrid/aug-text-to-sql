from database_connection.abstract_connector import AbstractDbConnector


def execution_accuracy(db_connector: AbstractDbConnector, pred_queries, real_queries):
    number_matches = 0

    for (pred_query, real_query) in zip(pred_queries, real_queries):
        pred_result = db_connector.execute(pred_query)
        real_result = db_connector.execute(real_query)
        if pred_result == real_result and pred_result is not None:
            number_matches += 1

    return number_matches / len(pred_queries)


def logical_form_accuracy(pred_queries, real_queries):
    return sum(map(lambda queries: queries[0] == queries[1], zip(pred_queries, real_queries))) / len(pred_queries)
