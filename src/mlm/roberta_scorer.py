import logging
from typing import List, Optional, Tuple, Union

import numpy as np

# MXNet-based
import gluonnlp as nlp
import gluonnlp.data.batchify as btf
from gluonnlp.model.bert import RoBERTaModel
import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon.data import SimpleDataset


from .loaders import Corpus
from .models import SUPPORTED_MLMS, SUPPORTED_LMS


class MLMScorerRoberta:
    """For models that need every token to be masked"""

    """A wrapper around a model which can score utterances
        """

    def __init__(
        self,
        model: Block,
        tokenizer,
        ctxs: List[mx.Context],
        eos: Optional[bool] = None,
        capitalize: Optional[bool] = None,
        wwm=False,
    ) -> None:

        self._wwm = wwm
        self._add_special = True

        self._model = model
        self._tokenizer = tokenizer
        self._ctxs = ctxs
        self._eos = eos
        self._capitalize = capitalize
        self._max_length = 1024

    def _apply_tokenizer_opts(self, sent: str) -> str:
        if self._eos:
            sent += "."
        if self._capitalize:
            sent = sent.capitalize()
        return sent

    def _corpus_to_data(
        self, corpus, split_size, ratio, num_workers: int, shuffle: bool = False
    ):

        # Turn corpus into a dataset
        dataset = self.corpus_to_dataset(corpus)

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            [sent_tuple[2] for sent_tuple in dataset],
            batch_size=split_size,
            ratio=ratio,
            num_shards=0,
            shuffle=shuffle,
        )

        logging.info(batch_sampler.stats())
        dataloader = nlp.data.ShardedDataLoader(
            dataset,
            pin_memory=True,
            batch_sampler=batch_sampler,
            batchify_fn=self._batchify_fn,
            num_workers=num_workers,
            thread_pool=True,
        )

        return dataset, batch_sampler, dataloader

    def _true_tok_lens(self, dataset):

        # Compute sum (assumes dataset is in order; skips are allowed)
        prev_sent_idx = None
        true_tok_lens = []
        for tup in dataset:
            curr_sent_idx = tup[0]
            valid_length = tup[2]
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                true_tok_lens.append(valid_length - 2)

        return true_tok_lens

    def _split_batch(self, batch):
        return zip(
            *[
                mx.gluon.utils.split_data(
                    batch_compo, len(self._ctxs), batch_axis=0, even_split=False
                )
                for batch_compo in batch
            ]
        )

    def score_sentences(
        self, sentences: List[str], **kwargs
    ) -> Union[List[float], List[List[float]], float]:
        corpus = Corpus.from_text(sentences)
        return self.score(corpus, **kwargs)[0]

    def _ids_to_masked(
        self, token_ids: np.ndarray
    ) -> List[Tuple[np.ndarray, List[int]]]:

        # Here:
        # token_ids = [2 ... ... 1012 3], where 2=[CLS], (optionally) 1012='.', 3=[SEP]

        token_ids_masked_list = []

        mask_indices = []
        if self._wwm:
            for idx, token_id in enumerate(token_ids):
                if self._tokenizer.is_first_subword(self._tokenizer.convert_ids_to_tokens(token_id)):
                    mask_indices.append([idx])
                else:
                    mask_indices[-1].append(idx)
        else:
            mask_indices = [[mask_pos] for mask_pos in range(len(token_ids))]

        # # We don't mask the [CLS], [SEP] for now for PLL
        if self._add_special:
            mask_indices = mask_indices[1:-1]
        else:
            mask_indices = mask_indices[1:]

        mask_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.mask_token)
        for mask_set in mask_indices:
            token_ids_masked = token_ids.copy()
            token_ids_masked[mask_set] = mask_token_id
            token_ids_masked_list.append((token_ids_masked, mask_set))

        return token_ids_masked_list

    def print_record(self, record):
        readable_sent = [self._tokenizer.convert_ids_to_tokens(tid) for tid in record[1]]
        logging.info(
            """
sent_idx = {},
text = {},
all toks = {},
masked_id = {}
        """.format(
                record[0], readable_sent, record[2], record[3], record[4]
            )
        )

    def corpus_to_dataset(self, corpus: Corpus) -> SimpleDataset:

        sents_expanded = []

        for sent_idx, sent in enumerate(corpus.values()):
            sent = self._apply_tokenizer_opts(sent)
            if self._add_special:
                tokenized = self._tokenizer(sent)
                tokens_original = (
                    [self._tokenizer.cls_token]
                    + tokenized["input_ids"]
                    + [self._tokenizer.sep_token]
                )
            else:
                tokens_original = [self._tokenizer.cls_token] + self._tokenizer(sent)
            ids_original = np.array(
                self._tokenizer.convert_tokens_to_ids(tokens_original)
            )

            # Enforce max length
            if len(ids_original) > self._max_length:
                logging.error(
                    "Line #{} is too long; will output score of 0 and omit in token counts (but not yet in word counts!)".format(
                        sent_idx + 1
                    )
                )
            else:
                ids_masked = self._ids_to_masked(ids_original)

                if self._wwm:
                    # TODO: Wasteful, but for now "deserialize" the mask set into individual positions
                    # The masks are already applied in ids
                    for ids, mask_set in ids_masked:
                        for mask_el, id_original in zip(
                            mask_set, ids_original[mask_set]
                        ):
                            sents_expanded.append(
                                (
                                    sent_idx,
                                    ids,
                                    len(ids_original),
                                    mask_el,
                                    [id_original],
                                    1,
                                )
                            )
                else:
                    sents_expanded += [
                        (
                            sent_idx,
                            ids,
                            len(ids_original),
                            mask_set,
                            ids_original[mask_set],
                            1,
                        )
                        for ids, mask_set in ids_masked
                    ]

        return SimpleDataset(sents_expanded)

    def score(
        self,
        corpus: Corpus,
        temp: float = 1.0,
        split_size: int = 2000,
        ratio: float = 0,
        num_workers: int = 10,
        per_token: bool = False,
    ) -> List[float]:

        ctx_cpu = mx.Context("cpu")

        # Turn corpus into a BERT-ready Dataset
        dataset = self.corpus_to_dataset(corpus)

        # Turn Dataset into Dataloader
        batchify_fn = btf.Tuple(
            btf.Stack(dtype="int32"),
            btf.Pad(
                pad_val=self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token),
                dtype="int32",
            ),
            btf.Stack(dtype="float32"),
            btf.Stack(dtype="float32"),
            btf.Stack(dtype="int32"),
            btf.Stack(dtype="float32"),
        )

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            [sent_tuple[2] for sent_tuple in dataset],
            batch_size=split_size,
            ratio=ratio,
            num_shards=0,
            shuffle=False,
        )

        logging.info(batch_sampler.stats())
        dataloader = nlp.data.ShardedDataLoader(
            dataset,
            pin_memory=True,
            batch_sampler=batch_sampler,
            batchify_fn=batchify_fn,
            num_workers=num_workers,
            thread_pool=True,
        )

        # Get lengths in tokens (assumes dataset is in order)
        prev_sent_idx = None
        true_tok_lens = []
        for (curr_sent_idx, _, valid_length, _, _, _) in dataset:
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                if self._add_special:
                    true_tok_lens.append(valid_length - 2)
                else:
                    true_tok_lens.append(valid_length - 1)

        # Compute scores (total or per-position)
        if per_token:
            if self._add_special:
                scores_per_token = [
                    [None] * (true_tok_len + 2) for true_tok_len in true_tok_lens
                ]
            else:
                scores_per_token = [
                    [None] * (true_tok_len + 1) for true_tok_len in true_tok_lens
                ]
        else:
            scores = np.zeros((len(corpus),))

        sent_count = 0
        batch_log_interval = 20

        batch_score_accumulation = 1
        batch_sent_idxs_per_ctx = [[] for ctx in self._ctxs]
        batch_scores_per_ctx = [[] for ctx in self._ctxs]
        batch_masked_positions_per_ctx = [[] for ctx in self._ctxs]

        def sum_accumulated_scores():
            for ctx_idx in range(len(self._ctxs)):
                for batch_sent_idxs, batch_scores, batch_masked_positions in zip(
                    batch_sent_idxs_per_ctx[ctx_idx],
                    batch_scores_per_ctx[ctx_idx],
                    batch_masked_positions_per_ctx[ctx_idx],
                ):
                    if per_token:
                        # Slow; only use when necessary
                        for batch_sent_idx, batch_score, batch_masked_position in zip(
                            batch_sent_idxs, batch_scores, batch_masked_positions
                        ):
                            scores_per_token[batch_sent_idx.asscalar()][
                                int(batch_masked_position.asscalar())
                            ] = batch_score.asscalar().item()
                    else:
                        np.add.at(
                            scores, batch_sent_idxs.asnumpy(), batch_scores.asnumpy()
                        )
                batch_sent_idxs_per_ctx[ctx_idx] = []
                batch_scores_per_ctx[ctx_idx] = []
                batch_masked_positions_per_ctx[ctx_idx] = []

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch_size = 0

            batch = zip(
                *[
                    mx.gluon.utils.split_data(
                        batch_compo, len(self._ctxs), batch_axis=0, even_split=False
                    )
                    for batch_compo in batch
                ]
            )

            for ctx_idx, (
                sent_idxs,
                token_ids,
                valid_length,
                masked_positions,
                token_masked_ids,
                normalization,
            ) in enumerate(batch):

                ctx = self._ctxs[ctx_idx]
                batch_size += sent_idxs.shape[0]
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                masked_positions = masked_positions.as_in_context(ctx).reshape(-1, 1)

                if isinstance(self._model, RoBERTaModel):
                    out = self._model(token_ids, valid_length, masked_positions)
                else:
                    segment_ids = mx.nd.zeros(shape=token_ids.shape, ctx=ctx)
                    out = self._model(
                        token_ids, segment_ids, valid_length, masked_positions
                    )

                # Get the probability computed for the correct token
                split_size = token_ids.shape[0]
                # out[0] contains the representations
                # out[1] is what contains the distribution for the masked

                # TODO: Manual numerically-stable softmax
                # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
                # Because we only need one scalar
                out = out[1].log_softmax(temperature=temp)

                # Save the scores at the masked indices
                batch_sent_idxs_per_ctx[ctx_idx].append(sent_idxs)
                out = out[
                    list(range(split_size)),
                    [0] * split_size,
                    token_masked_ids.as_in_context(ctx).reshape(-1),
                ]
                batch_scores_per_ctx[ctx_idx].append(out)
                batch_masked_positions_per_ctx[ctx_idx].append(masked_positions)

            # Ideally we'd accumulate the scores when possible, but something like the below won't work
            # > scores[sent_idxs] += out
            # See In[21] in https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html.
            # Hence, aggregation is done synchronously, every so often
            # (though batch_score_accumulation = 1 seems best, since bucketing is effective in reducing GPU disparity)
            if len(batch_sent_idxs_per_ctx[0]) == batch_score_accumulation:
                sum_accumulated_scores()

            # Progress
            sent_count += batch_size
            if (batch_id + 1) % batch_log_interval == 0:
                logging.info(
                    "{} sents of {}, batch {} of {}".format(
                        sent_count, len(dataset), batch_id + 1, len(batch_sampler)
                    )
                )

        # In case there are leftovers
        sum_accumulated_scores()

        if per_token:
            return scores_per_token, true_tok_lens
        else:
            return scores.tolist(), true_tok_lens