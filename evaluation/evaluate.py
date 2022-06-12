from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from database_connection.abstract_connector import AbstractDbConnector
from evaluation.methods import execution_accuracy, logical_form_accuracy
from evaluation.preprocess import get_sql, get_sql_substitution, get_variables, substitute_variables
from model import HydraNet
from model.hydra import SQLDataset


class Evaluator:
    def __init__(
        self,
        model: HydraNet,
        dataset: SQLDataset,
        raw_data: Dict,
        db_connector: AbstractDbConnector,
    ):
        self.model = model
        self.dataset = dataset
        self.raw_data = raw_data
        self.db_connector = db_connector

    def _infer_(self, inputs):
        self.model.eval()
        outputs = {}

        for start_idx in range(0, inputs["input_ids"].shape[0], HydraNet.BATCH_SIZE):
            input_tensor = {
                key: torch.from_numpy(inputs[key][start_idx : start_idx + HydraNet.BATCH_SIZE]).to(self.model.device)
                for key in HydraNet.MODEL_INPUT_KEYS
            }

            with torch.no_grad():
                model_output = self.model(**input_tensor)

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

        for pos in tqdm(dataset.pos):
            final_output = {}
            for k in model_outputs:
                final_output[k] = model_outputs[k][pos[0] : pos[1], :]
            dataset_outputs.append(final_output)

        assert len(dataset.input_features) == len(dataset_outputs)

        return dataset_outputs

    def _predict_sql_(self, dataset: SQLDataset, model_outputs=None):
        if model_outputs is None:
            model_outputs = self._infer_dataset_(dataset)

        sqls = []
        for input_feature, model_output in tqdm(zip(dataset.input_features, model_outputs)):
            agg, select, where, conditions = self._parse_output_(input_feature, model_output)

            conditions_with_value_texts = []
            for wc in where:
                _, op, vs, ve = conditions[wc]
                word_start, word_end = input_feature.subword_to_word[wc][vs], input_feature.subword_to_word[wc][ve]
                char_start = input_feature.word_to_char_start[word_start]
                char_end = len(input_feature.question)
                if word_end + 1 < len(input_feature.word_to_char_start):
                    char_end = input_feature.word_to_char_start[word_end + 1]
                value_span_text = input_feature.question[char_start:char_end].rstrip()
                conditions_with_value_texts.append((wc, op, value_span_text))

            where = "WHERE " + " AND ".join(conditions_with_value_texts)
            query = f"SELECT {select} FROM table {where}"  # TODO
            sqls.append(query)

        return sqls

    @classmethod
    def _get_where_num_(cls, output):
        relevant_prob = 1 - np.exp(output["column_func"][:, 2])
        where_num_scores = np.average(output["where_num"], axis=0, weights=relevant_prob)
        where_num = int(np.argmax(where_num_scores))

        return where_num

    @classmethod
    def _parse_output_(cls, input_feature, model_output):
        def get_span(i):
            offset = 0
            segment_ids = np.array(input_feature.segment_ids[i])
            for j in range(len(segment_ids)):
                if segment_ids[j] == 1:
                    offset = j
                    break

            value_start, value_end = (
                model_output["value_start"][i, segment_ids == 1],
                model_output["value_end"][i, segment_ids == 1],
            )
            l = len(value_start)
            sum_mat = value_start.reshape((l, 1)) + value_end.reshape((1, l))
            span = (0, 0)
            for cur_span, _ in sorted(np.ndenumerate(sum_mat), key=lambda x: x[1], reverse=True):
                if cur_span[1] < cur_span[0] or cur_span[0] == l - 1 or cur_span[1] == l - 1:
                    continue
                span = cur_span
                break

            return span[0] + offset, span[1] + offset

        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x: x[1], reverse=True)
        select = select_id_prob[0][0]
        agg = np.argmax(model_output["agg"][select, :])

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x: x[1], reverse=True)
        where_num = cls._get_where_num_(model_output)
        where = [i for i, _ in where_id_prob[:where_num]]
        conditions = {}
        for idx in set(where):
            span = get_span(idx)
            op = np.argmax(model_output["op"][idx, :])
            conditions[idx] = (idx, op, span[0], span[1])

        return agg, select, where, conditions

    def evaluate_preds(self, pred_queries):
        real_queries = get_sql(self.raw_data)
        exec_accuracy = execution_accuracy(self.db_connector, pred_queries, real_queries)
        logical_accuracy = logical_form_accuracy(pred_queries, real_queries)

        subst_real_queries = get_sql_substitution(self.raw_data)
        variables = get_variables(self.raw_data)
        subst_pred_queries = substitute_variables(variables, self.raw_data)

        match_accuracy = logical_form_accuracy(subst_pred_queries, subst_real_queries)

        return exec_accuracy, logical_accuracy, match_accuracy

    def evaluate(self):
        sqls = self._predict_sql_(self.dataset)
        self.evaluate_preds(sqls)
