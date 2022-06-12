from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from data_processing import SQLDataset, SQLFeaturizer
from database_connection.abstract_connector import AbstractDbConnector
from evaluation.methods import logical_form_accuracy
from model import RegSQLNet


class Evaluator:
    def __init__(self, model: RegSQLNet, db_connector: AbstractDbConnector, batch_size: int, verbose: bool):
        self.model = model
        self.db_connector = db_connector
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_(self, model_inputs):
        self.model.eval()
        outputs = {}

        for start_idx in tqdm(range(0, model_inputs["input_ids"].shape[0], self.batch_size)):
            batch = {
                key: model_inputs[key][start_idx : start_idx + self.batch_size].to(self.device)
                for key in ["input_ids", "attention_mask", "token_type_ids"]
            }

            with torch.no_grad():
                model_output = self.model(**batch)

            for key, out_tensor in model_output.items():
                if out_tensor is None:
                    continue
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(out_tensor.cpu().detach().numpy())

        for key in outputs:
            outputs[key] = np.concatenate(outputs[key], 0)

        return outputs

    def _infer_dataset_(self, dataset: SQLDataset):
        model_outputs = self._infer_(dataset.model_inputs)
        dataset_outputs = []

        for i in tqdm(range(dataset.model_inputs["input_ids"].shape[0])):
            final_output = {}
            for key in model_outputs:
                final_output[key] = model_outputs[key][i : i + 1, :]
            dataset_outputs.append(final_output)

        return model_outputs, dataset_outputs

    def _predict_sql_(self, dataset: SQLDataset, model_outputs=None):
        if model_outputs is None:
            model_outputs = self._infer_dataset_(dataset)

        result = {"agg": [], "select": [], "select_num": [], "where": [], "where_num": [], "op": [], "sql": []}
        for input_example, model_output in tqdm(zip(dataset.input_examples, model_outputs)):
            agg, select, select_num, where, where_num, conditions = self._parse_output_(model_output)

            select_blocks = [
                f"{input_example.cand_cols[idx].split()[1].lower()}alias0.{input_example.cand_cols[idx].split()[2].lower()}"
                for idx in select
            ]
            select_str = "select " + " , ".join(select_blocks)

            from_tables = list(map(lambda elem: f"{elem.lower()} AS {elem.lower()}alias0", input_example.tables))
            from_tables_str = "from " + " , ".join(from_tables)

            conditions_with_value_texts = []
            for wc in where:
                _, op = conditions[wc]
                value_span_text = "'text'"
                conditions_with_value_texts.append(
                    f"{f'{input_example.cand_cols[wc].split()[1].lower()}alias0.{input_example.cand_cols[wc].split()[2].lower()}'} {SQLFeaturizer.cond_ops[op]} {value_span_text}"
                )

            if len(conditions_with_value_texts) != 0:
                where_str = "where " + " and ".join(conditions_with_value_texts)
            else:
                where_str = ""

            query = f"{select_str} {from_tables_str} {where_str};"

            select_temp = np.zeros((len(input_example.cand_cols),))
            select_temp[select] = 1

            where_temp = np.zeros((len(input_example.cand_cols),))
            where_temp[where] = 1

            agg_temp = np.zeros((len(input_example.cand_cols),))
            agg_temp[select] = agg

            conditions_temp = np.zeros((len(input_example.cand_cols),))
            conditions_temp[list(conditions.keys())] = np.array(list(map(lambda elem: elem[1], conditions.values())))

            result["agg"].extend(agg_temp)
            result["select"].extend(select_temp)
            result["select_num"].extend([select_num] * len(input_example.cand_cols))
            result["where"].extend(where_temp)
            result["where_num"].extend([where_num] * len(input_example.cand_cols))
            result["op"].extend(conditions_temp)
            result["sql"].append(query)

        return result

    @classmethod
    def _get_arg_num_(cls, output, key_name: str):
        relevant_prob = 1 - np.exp(output["column_func"][:, 2])
        num_scores = np.average(output[key_name], axis=0, weights=relevant_prob)
        num = int(np.argmax(num_scores))

        return num

    @classmethod
    def _parse_output_(cls, model_output):
        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x: x[1], reverse=True)
        select_num = cls._get_arg_num_(model_output, "select_num")
        select = [i for i, _ in select_id_prob[:select_num]]
        agg = np.argmax(model_output["agg"][select, :])

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x: x[1], reverse=True)
        where_num = cls._get_arg_num_(model_output, "where_num")
        where = [i for i, _ in where_id_prob[:where_num]]
        conditions = {}

        for idx in set(where):
            op = np.argmax(model_output["op"][idx, :])
            conditions[idx] = (idx, op)

        return agg, select, select_num, where, where_num, conditions

    def evaluate_preds(self, prediction_result: Dict, dataset: SQLDataset):
        result = {}

        real_queries = dataset.sql_queries
        pred_queries = prediction_result["sql"]
        result["variable accuracy"] = logical_form_accuracy(pred_queries, real_queries)

        average_accuracy = 0
        for key in ["agg", "select_num", "select", "where_num", "where", "op"]:
            result[f"{key} accuracy"] = np.average(
                np.array(prediction_result[key]) == np.array(dataset.model_inputs[key])
            )
            average_accuracy += result[f"{key} accuracy"]
        result["average accuracy"] = average_accuracy

        return result

    def evaluate(self, dataset: SQLDataset) -> Dict[str, float]:
        model_outputs, dataset_outputs = self._infer_dataset_(dataset)
        result = self._predict_sql_(dataset, dataset_outputs)

        return self.evaluate_preds(result, dataset)

    def __to_device(self, data):
        data["input_ids"].to(self.device)
        data["attention_mask"].to(self.device)
        data["token_type_ids"].to(self.device)
