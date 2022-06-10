from database_connection.abstract_connector import AbstractDbConnector
from evaluation.methods import execution_accuracy, logical_form_accuracy
from evaluation.preprocess import get_sql, get_sql_substitution, get_variables, substitute_variables


def evaluate(db_connector: AbstractDbConnector, data, pred_queries):
    real_queries = get_sql(data)
    exec_accuracy = execution_accuracy(db_connector, pred_queries, real_queries)
    logical_accuracy = logical_form_accuracy(pred_queries, real_queries)

    subst_real_queries = get_sql_substitution(data)
    variables = get_variables(data)
    subst_pred_queries = substitute_variables(variables, data)

    match_accuracy = logical_form_accuracy(subst_pred_queries, subst_real_queries)

    return exec_accuracy, logical_accuracy, match_accuracy
