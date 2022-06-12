from torch import nn
from transformers import BertModel


class HydraNet(nn.Module):
    def __init__(
        self,
        pretrained_bert_model: str = "bert-base-cased",
        where_column_num: int = 4, # max col number in example
        agg_num: int = 6,
        op_num: int = 4,
        drop_rate: float = 0.2,
    ):
        super(HydraNet, self).__init__()
        self.base_model = BertModel.from_pretrained(pretrained_bert_model)

        self.dropout = nn.Dropout(drop_rate)

        bert_hid_size = self.base_model.config.hidden_size
        self.column_func = nn.Linear(bert_hid_size, 3)
        self.agg = nn.Linear(bert_hid_size, agg_num)
        self.op = nn.Linear(bert_hid_size, op_num)
        self.where_num = nn.Linear(bert_hid_size, where_column_num + 1)
        self.start_end = nn.Linear(bert_hid_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        agg=None,
        select=None,
        where=None,
        where_num=None,
        op=None,
        value_start=None,
        value_end=None,
    ):
        bert_output, pooled_output = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False
        )

        bert_output = self.dropout(bert_output)
        pooled_output = self.dropout(pooled_output)

        column_func_logit = self.column_func(pooled_output)
        agg_logit = self.agg(pooled_output)
        op_logit = self.op(pooled_output)
        where_num_logit = self.where_num(pooled_output)

        start_end_logit = self.start_end(bert_output)
        value_span_mask = attention_mask.to(dtype=bert_output.dtype)

        start_logit = start_end_logit[:, :, 0] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = start_end_logit[:, :, 1] * value_span_mask - 1000000.0 * (1 - value_span_mask)

        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            loss = cross_entropy(agg_logit, agg) * select.float()
            loss += bceloss(column_func_logit[:, 0], select.float())
            loss += bceloss(column_func_logit[:, 1], where.float()) # where[col_id] == 1 for true cond col id, we can make it a dict
            loss += bceloss(column_func_logit[:, 2], (1 - select.float()) * (1 - where.float()))
            loss += cross_entropy(where_num_logit, where_num)
            loss += cross_entropy(op_logit, op) * where.float()
            #TODO: what is this
            loss += cross_entropy(start_logit, value_start)
            loss += cross_entropy(end_logit, value_end)

        log_sigmoid = nn.LogSigmoid()

        return {
            "column_func": log_sigmoid(column_func_logit),
            "agg": agg_logit.log_softmax(1),
            "op": op_logit.log_softmax(1),
            "where_num": where_num_logit.log_softmax(1), # P(n_w | c_i, q)
            "value_start": start_logit.log_softmax(1),
            "value_end": end_logit.log_softmax(1),
            "loss": loss,
        }
