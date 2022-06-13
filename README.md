# Text-to-SQL Pipeline with Data Augmentation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI](https://github.com/Readrid/aug-text-to-sql/actions/workflows/github_ci.yml/badge.svg)

To run training with evaluation set up data paths in `config.yml` and run

```bash
python train.py
```

## Text-to-SQL Model

Text-to-SQL model was inspired by [HydraNet](https://arxiv.org/pdf/2008.04759.pdf). We use the simplest version of this model without Execution-Guided Decoding. [BERT](https://arxiv.org/pdf/1810.04805.pdf) is used as the first layer of the model.

## Paraphrase generation

GPT-NEO is used as a paraphrase generation model. 

## Contributors
* [Vladimir Fedorov](https://github.com/Readrid)
* [Ilya Smirnov](https://github.com/smirok)
* [Nikita Usoltsev](https://github.com/usoltsev37)
* [Anastasia Chaikova](https://github.com/achaikova)
