import unittest

from transformers import RobertaTokenizer, RobertaModel

from mlm.models.robbert import get_robbert_model
from mlm.roberta_scorer import MLMScorerRoberta
from mlm.scorers import MLMScorer
from mlm.models import get_pretrained
import mxnet as mx

ctxs = [mx.cpu()]  # or, e.g., [mx.gpu(0), mx.gpu(1)]


class RobertaTest(unittest.TestCase):
    def test_roberta_scoring(self):
        model, vocab, tokenizer = get_pretrained(ctxs, "roberta-base-en-cased")
        scorer = MLMScorer(model, vocab, tokenizer, ctxs)
        sentence_result = scorer.score_sentences(["Hello world!"])
        token_result = scorer.score_sentences(["Hello world!"], per_token=True)
        print(sentence_result)
        print(token_result)

        self.assertEqual(1, len(sentence_result))
        self.assertEqual(5, len(token_result[0]))

        self.assertEqual([-3.7631064113229513], sentence_result)
        self.assertEqual(
            [
                [
                    None,
                    -0.017823999747633934,
                    -2.315778970718384,
                    -1.4295034408569336,
                    None,
                ]
            ],
            token_result,
        )



    def test_roberta_scoring_new(self):
        # model, vocab, tokenizer = get_pretrained(ctxs, "roberta-base-en-cased")

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base")

        scorer = MLMScorerRoberta(model, tokenizer)
        sentence_result = scorer.score(["Hello world!"])
        token_result = scorer.score_sentences(["Hello world!"], per_token=True)
        print(sentence_result)
        print(token_result)

        self.assertEqual(1, len(sentence_result))
        self.assertEqual(5, len(token_result[0]))

        self.assertEqual([-3.7631064113229513], sentence_result)
        self.assertEqual(
            [
                [
                    None,
                    -0.017823999747633934,
                    -2.315778970718384,
                    -1.4295034408569336,
                    None,
                ]
            ],
            token_result,
        )

    def test_robbert_scoring(self):
        # model, vocab, tokenizer = get_pretrained(
        #     ctxs, "pdelobelle/robbert-v2-dutch-base"
        # )

        tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        model, vocab = get_robbert_model("pdelobelle/robbert-v2-dutch-base")

        scorer = MLMScorer(model, vocab, tokenizer, ctxs)
        sentence_result = scorer.score_sentences(["Hallo wereld!"])
        token_result = scorer.score_sentences(["Hallo wereld!"], per_token=True)
        print(sentence_result)
        # >> [-12.411025047302246]
        print(token_result)
        # >> [[None, -6.126738548278809, -5.501765727996826, -0.782496988773346, None]]

        self.assertEqual(1, len(sentence_result))
        self.assertLess(1, len(token_result[0]))


if __name__ == "__main__":
    unittest.main()
