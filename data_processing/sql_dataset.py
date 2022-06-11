from typing import List

import numpy as np
from torch.utils.data import Dataset

from featurizer import SQLFeaturizer
from utils import concat


class SQLDataset(Dataset):
    def __init__(self, sentences: List[str], sql_queries: List[str], featurizer: SQLFeaturizer, max_len: int):

        formatted_queries = featurizer.process_sql_queries(sql_queries)
        input_examples = featurizer.get_input_examples(sentences, formatted_queries)

        candidate_cols = [concat([row.type, row.table_name, row.col_name]) for i, row in featurizer.schema.iterrows()]

        self.size = len(candidate_cols) * len(input_examples)
        questions = np.repeat([ex.question for ex in input_examples], len(candidate_cols))

        self.encodings = featurizer.tokenizer(candidate_cols * len(input_examples), questions,
                                                   return_tensors="pt", max_length=max_len,
                                                   truncation=True, padding="max_length")

    def __getitem__(self, index):
        return {key: val[index] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encoding.input_ids)

