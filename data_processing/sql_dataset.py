from typing import List
from torch.utils.data import Dataset
import pandas as pd

from utils import preprocess_data, get_variables, substitute_variables, get_sql_substitution
from featurizer import SQLFeaturizer


class SQLDataset(Dataset):
    def __init__(self, json_data: List, schema: pd.DataFrame):
        data = preprocess_data(json_data)
        variables = get_variables(data)
        sql_queries = substitute_variables(variables, get_sql_substitution(data))

        self.featurizer = SQLFeaturizer(schema)

        format_queries = self.featurizer.process_sql_queries(sql_queries)
