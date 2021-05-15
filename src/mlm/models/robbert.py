"""
Most of this is copy pasted from gluonnlp because they only allow fixed number of models
"""

import os

from gluonnlp.base import get_home_dir
from gluonnlp.model.bert import (
    bert_hparams,
    BERTEncoder,
    RoBERTaModel,
    roberta_12_768_12_hparams,
)
import mxnet as mx
from gluonnlp.model.utils import _load_vocab, _load_pretrained_params


def get_roberta_model_modified(
    model_name=None,
    dataset_name=None,
    vocab=None,
    predefined_args=None,
    pretrained=True,
    ctx=mx.cpu(),
    use_decoder=True,
    output_attention=False,
    output_all_encodings=False,
    root=os.path.join(get_home_dir(), "models"),
    **kwargs
):
    """Any RoBERTa pretrained model.

    Parameters
    ----------
    model_name : str or None, default None
        Options include 'bert_24_1024_16' and 'bert_12_768_12'.
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        The supported datasets for model_name of either roberta_24_1024_16 and
        roberta_12_768_12 include 'openwebtext_ccnews_stories_books'.
    vocab : gluonnlp.vocab.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
        Note that
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert'
        do not include these parameters.
    output_attention : bool, default False
        Whether to include attention weights of each encoding cell to the output.
    output_all_encodings : bool, default False
        Whether to output encodings of all encoder cells.

    Returns
    -------
    RoBERTaModel, gluonnlp.vocab.Vocab
    """
    if predefined_args is None:
        if model_name in bert_hparams:
            predefined_args = bert_hparams[model_name]
        else:
            predefined_args = roberta_12_768_12_hparams

    mutable_args = ["use_residual", "dropout", "embed_dropout", "word_embed"]
    mutable_args = frozenset(mutable_args)
    assert all(
        (k not in kwargs or k in mutable_args) for k in predefined_args
    ), "Cannot override predefined model settings."
    predefined_args.update(kwargs)
    # encoder
    encoder = BERTEncoder(
        attention_cell=predefined_args["attention_cell"],
        num_layers=predefined_args["num_layers"],
        units=predefined_args["units"],
        hidden_size=predefined_args["hidden_size"],
        max_length=predefined_args["max_length"],
        num_heads=predefined_args["num_heads"],
        scaled=predefined_args["scaled"],
        dropout=predefined_args["dropout"],
        output_attention=output_attention,
        output_all_encodings=output_all_encodings,
        use_residual=predefined_args["use_residual"],
        activation=predefined_args.get("activation", "gelu"),
        layer_norm_eps=predefined_args.get("layer_norm_eps", None),
    )

    if dataset_name is not None:
        from ..vocab import Vocab

        bert_vocab = _load_vocab(dataset_name, vocab, root, cls=Vocab)
    else:
        bert_vocab = vocab

    # BERT
    net = RoBERTaModel(
        encoder,
        len(bert_vocab),
        units=predefined_args["units"],
        embed_size=predefined_args["embed_size"],
        embed_dropout=predefined_args["embed_dropout"],
        word_embed=predefined_args["word_embed"],
        use_decoder=use_decoder,
    )
    if pretrained:
        ignore_extra = not use_decoder
        _load_pretrained_params(
            net,
            model_name,
            dataset_name,
            root,
            ctx,
            ignore_extra=ignore_extra,
            allow_missing=False,
        )
    return net, bert_vocab


def get_robbert_model(
    model_name,
    dataset_name=None,
    vocab=None,
    pretrained=True,
    ctx=mx.cpu(),
    use_decoder=True,
    root=os.path.join(get_home_dir(), "models"),
    **kwargs
):
    return get_roberta_model_modified(
        model_name=model_name,
        vocab=vocab,
        dataset_name=dataset_name,
        pretrained=pretrained,
        ctx=ctx,
        use_decoder=use_decoder,
        root=root,
        **kwargs
    )
