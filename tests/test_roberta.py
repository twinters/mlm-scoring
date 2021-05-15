import unittest

from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
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
        # >> [-12.411025047302246]
        print(token_result)
        # >> [[None, -6.126738548278809, -5.501765727996826, -0.782496988773346, None]]

        self.assertEqual(1, len(sentence_result))
        self.assertLess(1, len(token_result[0]))


if __name__ == "__main__":
    unittest.main()
