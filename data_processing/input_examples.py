from typing import List, Tuple, Union, Dict

import pandas as pd

from data_processing.utils import type2canon, concat


class SQLQuery(object):
    def __init__(
        self,
        sel: List[Tuple[int, int]],
        conds: List[Tuple[int, int, Union[float, int, str]]],
        tables: List[str],
        schema: pd.DataFrame,
        col2id: Dict[str, int],
    ):
        self.sel = sel
        self.conds = conds
        self.tables = tables
        self.cand_cols = self.__get_candidate_columns(schema)

        self.__reset_ids(col2id)

    def __get_candidate_columns(self, schema):
        result = []
        for table_name in self.tables:
            cand_columns = schema[schema.table_name == table_name]
            for i, row in cand_columns.iterrows():
                result.append(concat([type2canon(row.type), row.table_name, row.col_name]))
        return result

    def __reset_ids(self, col2id):
        cand_cols_nums = list(map(lambda x: col2id[x], self.cand_cols))
        for i, (agg, col_num) in enumerate(self.sel):
            self.sel[i] = (agg, cand_cols_nums.index(col_num))
        for i, (col_num, op, val) in enumerate(self.conds):
            self.conds[i] = (cand_cols_nums.index(col_num), op, val)

    @property
    def where_num(self):
        return len(self.conds)

    @property
    def select_num(self):
        return len(self.sel)

    @property
    def cand_num(self):
        return len(self.cand_cols)


class InputExample(object):
    def __init__(
        self,
        question: str,
        sql_query: SQLQuery,  # label
    ):
        self.question = question

        self.cand_cols = sql_query.cand_cols

        self.select = [0] * sql_query.cand_num
        self.where = [0] * sql_query.cand_num
        self.where_num = [sql_query.where_num] * sql_query.cand_num
        self.select_num = [sql_query.select_num] * sql_query.cand_num
        self.agg = [0] * sql_query.cand_num
        self.op = [0] * sql_query.cand_num
        self.value_start = [0] * sql_query.cand_num
        self.value_end = [0] * sql_query.cand_num

        self.__fill_features(sql_query)

    def __fill_features(self, sql_query: SQLQuery):
        for col_agg, col_num in sql_query.sel:
            self.agg[col_num] = col_agg
            self.select[col_num] = 1

        for cond_col_num, cond_op_id, val in sql_query.conds:
            self.op[cond_col_num] = cond_op_id
            self.where[cond_col_num] = 1
            self.value_start[cond_col_num], self.value_end[cond_col_num] = self.__get_val_start_end(val)

    def __get_val_start_end(self, val: Union[float, int, str]) -> Tuple[int, int]:
        if isinstance(val, float) or isinstance(val, int):
            val = str(val)

        start = self.question.find(val)
        if start == -1:
            return 0, 0
        return start, start + len(val)
