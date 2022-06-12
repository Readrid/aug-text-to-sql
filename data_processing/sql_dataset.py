from typing import List

from torch.utils.data import Dataset

from data_processing.featurizer import SQLFeaturizer


class SQLDataset(Dataset):
    def __init__(self, sentences: List[str], sql_queries: List[str], featurizer: SQLFeaturizer):

        formatted_queries = featurizer.process_sql_queries(sql_queries)
        self.sql_queries = sql_queries
        self.input_examples = featurizer.get_input_examples(sentences, formatted_queries)
        self.model_inputs = featurizer.get_model_input(self.input_examples)

    def __getitem__(self, index):
        return {key: val[index] for key, val in self.model_inputs.items()}

    def __len__(self):
        return len(self.model_inputs["input_ids"])
