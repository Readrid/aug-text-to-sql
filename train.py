import pandas as pd
from omegaconf import OmegaConf

from data_processing import SQLFeaturizer, SQLDataset
from data_processing.utils import load_json, preprocess_data, get_sql_substitution, get_variables, substitute_variables
from model import Text2SQLTrainer, RegSQLNet


def main():
    config = OmegaConf.load("config.yml")

    column_names = ["table_name", "col_name", "is_primary_key", "is_foreign_key", "type", "other_ifo"]
    schema = pd.read_csv(config.dataset.csv_data, sep=", ", engine="python", names=column_names)

    featurizer = SQLFeaturizer(schema, config.max_len)
    dirty_json_data = load_json(config.dataset.json_data)
    train_dev_test_split = preprocess_data(dirty_json_data)
    subst_sql = get_sql_substitution(train_dev_test_split)
    variables = get_variables(train_dev_test_split)
    train_dev_test_split_queries = substitute_variables(variables, subst_sql)

    train_dataset = SQLDataset(
        train_dev_test_split["train"]["sentences"],
        train_dev_test_split_queries["train"],
        featurizer,
    )
    eval_dataset = SQLDataset(
        train_dev_test_split["dev"]["sentences"],
        train_dev_test_split_queries["dev"],
        featurizer,
    )
    test_dataset = SQLDataset(
        train_dev_test_split["test"]["sentences"],
        train_dev_test_split_queries["test"],
        featurizer,
    )

    model = RegSQLNet(**config.regsql_cnf)
    trainer = Text2SQLTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, **config.train_cnf)
    trainer.train()
    trainer.eval(test_dataset)


if __name__ == "__main__":
    main()
