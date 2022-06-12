from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import pandas as pd
import sqlparse
import torch
from transformers import AutoTokenizer

from data_processing.input_examples import InputExample, SQLQuery
from data_processing.utils import concat, type2canon

QueryRepresentation = Dict[str, Union[List[Tuple[int, int]], List[Tuple[int, int, Union[float, int, str]]], List[str]]]


class SQLFeaturizer(object):
    agg_ops = ["", "max", "min", "count", "sum", "avg"]
    cond_ops = ["=", ">", "<", ">=", "<=", "OP"]

    def __init__(self, schema: pd.DataFrame, max_len: int):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.schema = schema
        self.max_len = max_len

        self.col2id = dict()
        self.col2type = dict()
        for i, row in self.schema.iterrows():
            self.col2id[concat([type2canon(row.type), row.table_name, row.col_name])] = i
            self.col2type[concat([row.table_name, row.col_name])] = type2canon(row.type)
        self.agg2id = lambda x: self.agg_ops.index(x)
        self.cond_ops2id = lambda x: self.cond_ops.index(x)

    def get_model_input(self, examples: List[InputExample], include_labels=True):
        model_inputs = {key: [] for key in ["input_ids", "attention_mask", "token_type_ids"]}
        if include_labels:
            for k in ["agg", "select", "select_num", "where_num", "where", "op", "value_start", "value_end"]:
                model_inputs[k] = []

        for example in examples:
            tokenized = self.tokenizer(
                np.repeat(example.question, len(example.cand_cols)).tolist(),
                example.cand_cols,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
            )
            model_inputs["input_ids"].extend(tokenized["input_ids"])
            model_inputs["attention_mask"].extend(tokenized["attention_mask"])
            model_inputs["token_type_ids"].extend(tokenized["token_type_ids"])

            if include_labels:
                model_inputs["agg"].extend(example.agg)
                model_inputs["select"].extend(example.select)
                model_inputs["select_num"].extend(example.select_num)
                model_inputs["where_num"].extend(example.where_num)
                model_inputs["where"].extend(example.where)
                model_inputs["op"].extend(example.op)
                model_inputs["value_start"].extend(example.value_start)
                model_inputs["value_end"].extend(example.value_end)

        model_inputs["input_ids"] = torch.LongTensor(model_inputs["input_ids"])
        model_inputs["attention_mask"] = torch.LongTensor(model_inputs["attention_mask"])
        model_inputs["token_type_ids"] = torch.LongTensor(model_inputs["token_type_ids"])

        return model_inputs

    def get_input_examples(self, questions: List[str], sql_queries: List[SQLQuery]) -> List[InputExample]:
        return [InputExample(question, query) for question, query in zip(questions, sql_queries)]

    def process_sql_queries(self, sql_queries: List[str]) -> List[SQLQuery]:
        return [self.__process_sql_query(sql_q) for sql_q in sql_queries]

    def __process_sql_query(self, sql_example: str, debug=False) -> SQLQuery:
        sql_example = sql_example.replace(" ,", ",")
        if debug:
            print(sql_example)
        formatted_sql = sqlparse.format(sql_example, reindent=True, keyword_case="upper")
        if debug:
            print(formatted_sql)
        formatted_sql = list(map(lambda x: x[:-1] if x.endswith(",") else x, formatted_sql.split("\n")))
        formatted_sql[-1] = formatted_sql[-1][:-2]

        tmp = []
        for line in formatted_sql:
            if "," in line:
                nl = line.split(", ")
                for i in range(1, len(nl)):
                    nl[i] = "       " + nl[i]
                tmp += nl
            else:
                tmp.append(line)
        formatted_sql = tmp
        if debug:
            print(formatted_sql)

        mode = None
        sel = []
        conds = []
        tables = set()

        for line in formatted_sql:
            if line.startswith("GROUP BY") or line.startswith("ORDER BY") or line.startswith("HAVING"):
                mode = None
            elif mode == "WHERE" or line.startswith("WHERE"):
                mode = "WHERE"
                conds.append(self.__process_cond(line))
            elif mode == "FROM" or line.startswith("FROM"):
                mode = "FROM"
                tables.add(self.__process_from(line))
            elif mode == "SELECT" or line.startswith("SELECT"):
                mode = "SELECT"
                sel.append(self.__process_select_line(line))

        return SQLQuery(sel, conds, list(tables), self.schema, self.col2id)

    def __process_select_line(self, line: str) -> Tuple[int, Optional[int]]:
        column = line[7:]

        splitted = column.split("(")
        splitted[0] = splitted[0].lower().strip()
        agg_op_id = self.agg2id("")
        if splitted[0] in self.agg_ops and len(splitted) > 1:
            col_num = 1
            col_slice = -1
            if splitted[1].startswith("DISTINCT"):
                col_num = 2
                col_slice = -2

            agg_op_id = self.agg2id(splitted[0])
            column = splitted[col_num][:col_slice]

        return (agg_op_id, self.col2id[self.__parse_col_info(column)])

    def __process_cond(self, line: str) -> Tuple[Optional[int], int, Union[str, int, float]]:
        cond = line[6:]
        split = cond.split()
        col_id = self.col2id[self.__parse_col_info(split[0])]
        op_id = self.cond_ops2id(split[1])

        val = " ".join(split[2:])
        if val[0] != '"' and val[-1] != '"':
            try:
                val = int(val)
            except:
                try:
                    val = float(val)
                except:
                    val = self.__parse_col_info(val)
        else:
            val = val[1:-1]

        return (col_id, op_id, val)

    def __process_from(self, line: str) -> str:
        table_with_alias = line[5:]
        table_name = table_with_alias.split(" AS ")[0]
        return table_name.lower()

    def __parse_col_info(self, column: str) -> int:
        column = column.split(".")
        table_name = column[0].split("alias")[0]
        column_name = column[1]
        type_name = self.col2type[concat([table_name, column_name])]
        return concat([type_name, table_name, column_name])
