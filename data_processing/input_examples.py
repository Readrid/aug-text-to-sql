from typing import List, Tuple, Union, Set

import pandas as pd


class SQLQuery(object):
    def __init__(
        self, sel: List[Tuple[int, int]], conds: List[Tuple[int, int, Union[float, int, str]]], schema: pd.DataFrame
    ):
        self.sel = sel
        self.conds = conds
        self.tables = self.__get_tables(schema)

    def __get_tables(self, schema: pd.DataFrame) -> Set[str]:
        tables = set()
        for col in self.sel:
            tables.add(schema.iloc[col[1]].table_name)
        for col in self.conds:
            tables.add(schema.ilocp[col[0]].table_name)
        return tables


class InputExample(object):
    def __init__(
        self,
        question: str,
        sql_query: SQLQuery,  # label
    ):
        self.question = question
        self.sql_query = sql_query
