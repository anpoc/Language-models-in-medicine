import torch
import math
import sys

from typing import Union
from numpy import diff, repeat
from copy import deepcopy
from datasets import Dataset


class PairTokenizer():
    def __init__(self, tokenizer, tokenizer_kwargs:dict):
        """Instantiate a PairTokenizer, a tokenizer for pairs of inputs.

        Args:
            tokenizer: tokeinzer.
            tokenizer_kwargs (dict): tokenizer configuration arguments.
        """
        self.tokenizer = tokenizer
        self.kwargs = deepcopy(tokenizer_kwargs)
        self.tail_max_len = 0
        self.tail_tokens = {'input_ids': [], 'attention_mask': []}
        self.extra_in_between = len(tokenizer('a', 'b').input_ids) - 5


    def tokenize_tail(self, texts:list):
        """Tokenize the second text of the pair of texts.

        Args:
            texts (list): second texts of a pair of texts.
        """
        tail_tokens = tokenize_batch(texts, self.tokenizer, {})
        self.postprocess_tail(tail_tokens)
        assert(self.tail_max_len < self.kwargs['max_length']), \
            "The size of the labels' tokens exceeds the max input size of the model."
        self.kwargs['max_length'] -= self.tail_max_len


    def postprocess_tail(self, tokens:dict):
        """Postprocess the tokens corresponding to the second text of a pair of texts.
            Postprocessing includes adding and/or removing tokens and updating.

        Args:
            tokens (dict): tokens.
        """
        for idx in range(len(tokens['input_ids'])):
            if self.extra_in_between:
                tokens['input_ids'][idx] = tokens['input_ids'][idx][-1:] + tokens['input_ids'][idx][1:]
            else:
                tokens['input_ids'][idx] = tokens['input_ids'][idx][1:]
                tokens['attention_mask'][idx] = tokens['attention_mask'][idx][1:]
        self.tail_max_len = max(tokens['length']) + (self.extra_in_between - 1)
        self.tail_tokens.update(tokens)


    def tokenize_complete(self, texts:list) -> dict:
        """Tokenize the complete pair of texts.

        Args:
            texts (list): first texts of a pair of texts.

        Returns:
            dict: tokens of the pair of texts.
        """
        head_tokens = tokenize_batch(texts, self.tokenizer, self.kwargs)
        model_input = {
            'input_ids': [head + tail for head in head_tokens['input_ids'] 
                for tail in self.tail_tokens['input_ids']],
            'attention_mask': [head + tail for head in head_tokens['attention_mask'] 
                for tail in self.tail_tokens['attention_mask']],
        }
        if 'overflow_to_sample_mapping' in head_tokens.keys():
            model_input['overflow_to_sample_mapping'] = list(repeat(
                head_tokens['overflow_to_sample_mapping'], len(self.tail_tokens['input_ids'])))

        return self.tokenizer.pad(model_input)


class GenTokenizer():
    def __init__(self, tokenizer, tokenizer_kwargs:dict, start_indexes:list):
        """Instantiate a GenTokenizer, a tokenizer for generative tasks.

        Args:
            tokenizer: tokenizer.
            tokenizer_kwargs (dict): tokenizer configuration arguments.
            start_indexes (list): indices from which the generation process starts.
        """
        self.tokenizer = tokenizer
        self.kwargs = deepcopy(tokenizer_kwargs)
        self.kwargs['return_tensors'] = 'pt'

        # Checkig existing or defining special required tokens
        ## BOS token
        if (self.tokenizer.bos_token is None) and (-1 in start_indexes):
            start_indexes.remove(-1)
            assert(len(start_indexes) != 0), 'Model should have a bos_token to use -1 as start_indexes.'
        ## PAD token
        if self.tokenizer.pad_token is None:
            self.define_special_token('pad_token')

        # Whether the bos token should be included
        self.include_bos_token = -1 in start_indexes
        self.kwargs['max_length'] -= int(self.include_bos_token)


    def define_special_token(self, token_name:str):
        """Define a special token.

        Args:
            token_name (str): name of the special token.
        """
        special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
        assert ((len(special_tokens) > 0) or (token_name == 'pad_token')), 'Model must have at '\
            f'least one special token to use as {token_name}.Please use a different model.'
        if len(special_tokens) == 0:
            self.tokenizer.add_special_tokens({token_name: '.'})
        else:
            self.tokenizer.add_special_tokens({token_name: special_tokens[0]})


    def tokenize(self, texts:list) -> dict:
        """Tokenize some texts.

        Args:
            texts (list): texts to tokenize.

        Returns:
            dict: tokens.
        """
        tokenized_data = tokenize_batch(texts, self.tokenizer, self.kwargs)
        if self.include_bos_token:
            assert(self.tokenizer.padding_side != 'left'), 'Modify the padding side: model skipped.'
            t = torch.ones((tokenized_data['input_ids'].size(dim=0), 1), dtype=torch.int64)
            tokenized_data['input_ids'] = torch.cat([t * self.tokenizer.bos_token_id, 
                tokenized_data['input_ids']], dim=1)
            tokenized_data['attention_mask'] = torch.cat([t, tokenized_data['attention_mask']], dim=1)
        return tokenized_data


class MCATokenizer():
    def __init__(self, tokenizer, tokenizer_kwargs:dict):
        """Instantiate a MCATokenizer, a tokenizer for multiple-choice question-answering tasks.

        Args:
            tokenizer: tokenizer.
            tokenizer_kwargs (dict): tokenizer configuration arguments.
        """
        self.tokenizer = tokenizer
        self.kwargs = deepcopy(tokenizer_kwargs)
        self.prompt = []

        # Checkig existing or defining special required tokens
        ## PAD token
        if self.tokenizer.pad_token is None:
            self.define_special_token('pad_token')

        aux_w = tokenizer('the')['input_ids']
        aux_wo = tokenizer('the', add_special_tokens=False)['input_ids']
        self.use_bos_token = aux_w[0] != aux_wo[0]
        self.use_eos_token = aux_w[-1] != aux_wo[-1]


    def define_special_token(self, token_name:str):
        """Define a special token.

        Args:
            token_name (str): name of the special token.
        """
        special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
        assert ((len(special_tokens) > 0) or (token_name == 'pad_token')), 'Model must have at '\
            f'least one special token to use as {token_name}.Please use a different model.'
        if len(special_tokens) == 0:
            self.tokenizer.add_special_tokens({token_name: '.'})
        else:
            self.tokenizer.add_special_tokens({token_name: special_tokens[0]})


    def tokenize(self, texts:list) -> dict:
        """Tokenize some texts.

        Args:
            texts (list): texts to tokenize.

        Returns:
            dict: tokens.
        """
        tokenized_data = tokenize_batch(texts, self.tokenizer, self.kwargs).input_ids
        bos = [self.tokenizer.bos_token_id] if self.use_bos_token else []
        eos = [self.tokenizer.eos_token_id] if self.use_eos_token else []
        prefix = self.prompt[0] if len(self.prompt) > 1 else []
        suffix = self.prompt[-1] if len(self.prompt) > 0 else []
        tokenized_data = dict(input_ids = list(map(lambda x: bos + prefix + x + suffix + eos, tokenized_data)))

        return self.tokenizer.pad(tokenized_data)


    def set_prompt(self, prompt:str):
        """Set a prompt.

        Args:
            prompt (str): prompt.
        """        
        self.prompt, plen = list(map(tokenize_batch(list(filter(None, prompt.split('{report}'))), 
            self.tokenizer, {'add_special_tokens': False}).get, ['input_ids', 'length']))
        self.kwargs['max_length'] -= sum(plen)


def update_sample_mapping(dataset:datasets.Dataset) -> datasets.Dataset:
    """Update the mapping between each element and its sample ID, specially important for samples
        divided into multiple subsamples.

    Args:
        dataset (datasets.Dataset): dataset.

    Returns:
        datasets.Dataset: dataset with mapping to sample IDs.
    """
    if'overflow_to_sample_mapping' in dataset.features:
        new_ids = [0] + list((diff(dataset['overflow_to_sample_mapping']) != 0).cumsum())
        dataset = dataset.remove_columns('overflow_to_sample_mapping')
        dataset = dataset.add_column('overflow_to_sample_mapping', new_ids)
    return dataset


def tokenize_batch(text_batch:list, tokenizer, kwargs:dict) -> dict:
    """Tokenize a batch of texts.

    Args:
        text_batch (list): batch of texts.
        tokenizer: _tokenizer.
        kwargs (dict): tokenizer configuration arguments.

    Returns:
        dict: tokens.
    """
    tokenized = tokenizer(text_batch, **kwargs, return_length=True)
    if not(kwargs.get('truncation', False)) and ('max_length' in kwargs.keys()):
        assert(max(tokenized['length']) < kwargs['max_length']), \
            "Model's max input len exceeded and truncation set to 'False': model skipped."
    return tokenized


def detokenize_batch(tokens_batch:Union[list, np.ndarray], tokenizer) -> list:
    """Detokenize a bacth of tokens.

    Args:
        token_batch (Union[list, np.ndarray]): batch of tokens.
        tokenizer: tokenizer.

    Returns:
        list: texts.
    """
    return tokenizer.batch_decode(token_batch, skip_special_tokens=True)