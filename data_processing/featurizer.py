import pandas as pd

from typing import Optional, Tuple, Union, List, Dict

import sqlparse
from transformers import AutoTokenizer

from utils import concat

from input_examples import InputExample, SQLQuery

QueryRepresentation = Dict[str, Union[List[Tuple[int, int]], List[Tuple[int, int, Union[float, int, str]]]]]


class SQLFeaturizer(object):
    agg_ops = ["", "max", "min", "count", "sum", "avg"]
    cond_ops = ["=", ">", "<", ">=", "<=", "OP"]

    def __init__(self, schema: pd.DataFrame):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.schema = schema

        self.col2id = dict()
        self.col2type = dict()
        for i, row in self.schema.iterrows():
            self.col2id[concat([SQLFeaturizer.__type2canon(row.Type), row.TableName, row.FieldName])] = i
            self.col2type[concat([row.TableName, row.FieldName])] = SQLFeaturizer.__type2canon(row.Type)
        self.agg2id = lambda x: self.agg_ops.index(x)
        self.cond_ops2id = lambda x: self.cond_ops.index(x)

    def get_input_examples(self, questions: List[str], sql_queries: List[QueryRepresentation]) -> List[InputExample]:
        result = []
        for question, query in zip(questions, sql_queries):
            result.append(InputExample(question, SQLQuery(**query, schema=self.schema)))
        return result

    def process_sql_queries(self, sql_queries: List[str]) -> List[QueryRepresentation]:
        result = []
        for i, sql_q in enumerate(sql_queries):
            sel, conds = self.__process_sql_query(sql_q)
            result.append({"sel": sel, "conds": conds})

        return result

    def __process_sql_query(
        self, sql_example: str, debug=False
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, Union[float, int, str]]]]:
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

        for line in formatted_sql:
            if line.startswith("GROUP BY") or line.startswith("ORDER BY") or line.startswith("HAVING"):
                mode = None
            elif mode == "WHERE" or line.startswith("WHERE"):
                mode = "WHERE"
                conds.append(self.__process_cond(line))
            elif mode == "FROM" or line.startswith("FROM"):
                mode = "FROM"
            elif mode == "SELECT" or line.startswith("SELECT"):
                mode = "SELECT"
                sel.append(self.__process_select_line(line))
        return sel, conds

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

    def __parse_col_info(self, column: str) -> int:
        column = column.split(".")
        table_name = column[0].split("alias")[0]
        column_name = column[1]
        type_name = self.col2type[concat([table_name, column_name])]
        return concat([type_name, table_name, column_name])

    @staticmethod
    def __type2canon(type_name):
        return type_name.split("(")[0]
