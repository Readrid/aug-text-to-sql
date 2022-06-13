from bert_score import score
from rouge_score import rouge_scorer

from paraphrase_generation.data.utils import read_json


def calc_bert_score(references, predictions):
    result = score(predictions, references, lang="en", verbose=True)
    return (result[0].mean(), result[1].mean(), result[2].mean())


def calc_rouge_score(references, predictions):
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    scorer = rouge_scorer.RougeScorer([ROUGE1, ROUGE2, ROUGEL], use_stemmer=True)
    total_scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]

    rouge1 = {"precision": 0, "recall": 0, "fmeasure": 0}
    rouge2 = {"precision": 0, "recall": 0, "fmeasure": 0}
    rougeL = {"precision": 0, "recall": 0, "fmeasure": 0}

    for sent_score in total_scores:
        rouge1["precision"] += sent_score[ROUGE1].precision
        rouge1["recall"] += sent_score[ROUGE1].recall
        rouge1["fmeasure"] += sent_score[ROUGE1].fmeasure

        rouge2["precision"] += sent_score[ROUGE2].precision
        rouge2["recall"] += sent_score[ROUGE2].recall
        rouge2["fmeasure"] += sent_score[ROUGE2].fmeasure

        rougeL["precision"] += sent_score[ROUGEL].precision
        rougeL["recall"] += sent_score[ROUGEL].recall
        rougeL["fmeasure"] += sent_score[ROUGEL].fmeasure

    rouge1["precision"] /= len(references)
    rouge1["recall"] /= len(references)
    rouge1["fmeasure"] /= len(references)
    rouge2["precision"] /= len(references)
    rouge2["recall"] /= len(references)
    rouge2["fmeasure"] /= len(references)
    rougeL["precision"] /= len(references)
    rougeL["recall"] /= len(references)
    rougeL["fmeasure"] /= len(references)

    return rouge1, rouge2, rougeL


if __name__ == "__main__":
    path = "../evaluation/eval_yelp_real2_copy.json"
    data = read_json(path)

    predictions = [s["paraphrases"][0] for s in data]
    references = [s["sentence"] for s in data]

    print(f"Bert_score: precision, recall, fmeasure\n{calc_bert_score(references=references, predictions=predictions)}")
    rouges = calc_rouge_score(references=references, predictions=predictions)
    print(f"Rouge1 \n{rouges[0]}")
    print(f"Rouge2 \n{rouges[1]}")
    print(f"Rougel \n{rouges[2]}")
