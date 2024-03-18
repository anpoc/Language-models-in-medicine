import os, sys, logging
import torch

from typing import Optional
from copy import deepcopy
from time import time
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, \
    AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, \
    OPTForCausalLM, BioGptTokenizer, BioGptForCausalLM, T5Tokenizer, T5ForConditionalGeneration, \
    AutoModelForSeq2SeqLM, DisjunctiveConstraint

from encoderUtils import Encoder, aggregate_update_embedding, softmax, cosine_similarity
from decoderUtils import Decoder
from dataUtils import *
from performanceUtils import *
from tokenizationUtils import *


global multilabel, device, file_path


def get_model_tokenizer(model_name:str, task_name:str):
    """Get the specified model and its tokenizer from HugginFace, along with some of its attributes.

    Args:
        model_name (str): name of the model.
        task_name (str): task for which the model is employed.

    Returns:
        HugginFace model and tokenizer, along with the name and maximum input length of the model.
    """
    if task_name == 'nli':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task_name in ['complete', 'decode']:
        if ('llama' in model_name) or ('alpaca' in model_name):
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, 
                torch_dtype=torch.float16 if '13b' in model_name else torch.float)
        elif ('gpt-2' in model_name) or ('gpt2' in model_name):
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        elif 'biogpt' in model_name.lower():
            tokenizer = BioGptTokenizer.from_pretrained(model_name)
            model = BioGptForCausalLM.from_pretrained(model_name)
        elif 't5' in model_name.lower():
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name, 
                torch_dtype=torch.float16 if 'xxl' in model_name else torch.float,
                from_flax='ClinicalT5' in model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if 'galactica' in model_name: model = OPTForCausalLM.from_pretrained(model_name)
            elif 'T0' in model_name: model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                torch_dtype=torch.float if '3B' in model_name else torch.float16)
            else: model = AutoModelForCausalLM.from_pretrained(model_name, 
                trust_remote_code=(('mpt-7b' in model_name) or ('falcon' in model_name)))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    model.eval()

    max_model_len = tokenizer.model_max_length
    # If model max length is not specified as a tokenizer attribute, try to extract it from the 
    # model's configuration. Set to 512 if not found.
    if max_model_len == 1e30:
        max_model_len = getattr(AutoConfig.from_pretrained(model_name), 'max_position_embeddings', 512)
    model_name = model_name.split('/')[-1]

    return model, tokenizer, model_name, max_model_len


def encode_batch(batch:dict, model) -> dict:
    """Encode a text batch.

    Args:
        batch (dict): input_ids and attention_mask of the text batch to process.
        model: model to encode the batch.

    Returns:
        dict: batch embeddings.
    """
    model_input = {
        'input_ids': torch.tensor(batch['input_ids']).to(device=device),
        'attention_mask': torch.tensor(batch['attention_mask']).to(device=device)
    }
    with torch.no_grad():
        embedding = model(model_input)

    return {'embedding': embedding.detach().cpu().numpy()}


def encode(dataset, labels:list, model, tokenizer, model_name:str, batch_size:list, 
    tokenizer_kwargs:dict, dataset_name:str, pool_funcs:list=['CLS_pool', 'max_pool', 'avg_pool'], 
    template:tuple=('', {'template_str': '{label}'})) -> list:
    """Perform text classification of a dataset via contextual embedding similarity.

    Args:
        dataset: dataset to process.
        labels (list): name of the classes/labels.
        model: model to perform the task.
        tokenizer: tokenizer of the model.
        model_name (str): name of the model.
        batch_size (list): list with batch sizes for tokenizer, model and general dataset processing.
        tokenizer_kwargs (dict): arguments to be passed to the tokenizer.
        dataset_name (str): name of the dataset.
        pool_funcs (list, optional): pooling strategies to run. Defaults to ['CLS_pool', 'max_pool', 
            'avg_pool'].
        template (tuple, optional): template to use for the encoding of the labels. Defaults to 
            ('', {'template_str': '{label}'}).

    Returns:
        list: contextual embedding similarity results.
    """
    # Variables
    results:list = []
    file_name = f'{file_path}temp/encode_{dataset_name}_{model_name}_' + '{x}'

    # Getting the template and applying it to the labels
    template_ID, template_str = str(template[0]), template[1]['template_str']
    labels = [template_str.format(label=label) for label in labels]

    # Running the different pooling strategies
    for pool_func in pool_funcs:
        encoder = Encoder(model, pool_func).to(device=device)
        start_time = time()
        pending_flag_ds, encoded_dataset = check_existence_load(file_name.format(x=pool_func), 
            constraint=dataset.num_rows-1)
        if pending_flag_ds:
            # Tokenizing, encoding and saving
            encoded_dataset = dataset.map(lambda x: tokenize_batch(x['feature'], tokenizer, 
                tokenizer_kwargs), batched=True, batch_size=batch_size[0], 
                remove_columns=dataset.column_names)
            encoded_dataset = update_sample_mapping(encoded_dataset)
            encoded_dataset = encoded_dataset.map(lambda x: encode_batch(x, encoder), batched=True, 
                batch_size=batch_size[-1])
            encoded_dataset.save_to_disk(file_name.format(x=pool_func))
        end_time = time()
        dataset = aggregate_update_embedding(pool_func, dataset, encoded_dataset, 'embedding')

        # Running encoding of labels
        tokenized_labels = tokenize_batch(labels, tokenizer, tokenizer_kwargs)
        encoded_labels = encode_batch(tokenized_labels, encoder)

        # Calculating results
        scores = cosine_similarity(torch.tensor(dataset['embedding']).to(device=device), 
            torch.tensor(encoded_labels['embedding']).to(device=device))

        results.append(
            [f'{model_name}_{pool_func}_{template_ID}', dataset_name] + 
            measure_classification(dataset['target'], scores, multilabel=multilabel) + 
            [round(end_time - start_time, 8)]
        )

        dataset = dataset.remove_columns('embedding')

    return results


def nli_batch(batch:dict, model) -> dict:
    """Get the logits after running natural language inference (NLI) on a batch.

    Args:
        batch (dict): input_ids and attention_mask of the text batch to process.
        model: model to use for NLI.

    Returns:
        dict: batch logits.
    """
    model_input = {
        'input_ids': torch.tensor(batch['input_ids']).to(device=device),
        'attention_mask': torch.tensor(batch['attention_mask']).to(device=device)
    }
    with torch.no_grad():
        logits = model(**model_input).logits

    return {'logits': logits.detach().cpu().numpy()}


def nli(dataset, labels:list, model, tokenizer, model_name:str, batch_size:list, 
    tokenizer_kwargs:dict, dataset_name:str, template:tuple=('', {'template_str': '{label}', 
    'incl_neg': 0, 'neg_str': ''})) -> list:
    """Perform text classification of a dataset via natural language inference (NLI).

    Args:
        dataset: dataset to process.
        labels (list): name of the classes/labels.
        model: model to perform the task.
        tokenizer: tokenizer of the model.
        model_name (str): name of the model.
        batch_size (list): list with batch sizes for tokenizer, model and general dataset processing.
        tokenizer_kwargs (dict): arguments to be passed to the tokenizer.
        dataset_name (str): name of the dataset.
        template (tuple, optional): template to use for the hyphotesis generation. Defaults to 
            ('', {'template_str': '{label}', 'incl_neg': 0, 'neg_str': ''}).

    Returns:
        list: NLI results.
    """
    # Variables
    results:list = []
    file_name:str = f'{file_path}temp/nli_{dataset_name}_{model_name}_{template[0]}'
    tokenizerPair = PairTokenizer(tokenizer, tokenizer_kwargs)
    model.to(device=device)

    # Getting the template and applying it to the labels
    template_ID, template_str = str(template[0]), template[1]['template_str']
    labels = [template_str.format(label=label) for label in labels]
    labels += [template[1]['neg_str']] if template[1]['incl_neg'] else []

    # Processing
    start_time = time()
    tokenizerPair.tokenize_tail(labels)
    logits_dataset = dataset.map(lambda x: tokenizerPair.tokenize_complete(x['feature']), 
        batched=True, batch_size=batch_size[0], remove_columns=dataset.column_names)
    logits_dataset = update_sample_mapping(logits_dataset)
    logits_dataset = logits_dataset.map(lambda x: nli_batch(x, model), batched=True, 
        batch_size=batch_size[-1])
    logits_dataset.save_to_disk(file_name)
    end_time = time()

    # Calculating results
    if multilabel:
        scores = softmax(torch.tensor(logits_dataset['logits'])[:,[0,2]].to(device=device))[:,-1]\
            .view(-1, len(labels)).cpu().numpy()
    else:
        scores = softmax(torch.tensor(logits_dataset['logits'])[:,2].view(-1, len(labels))\
            .to(device=device)).cpu().numpy()
    if 'overflow_to_sample_mapping' in logits_dataset.features:
        scores = update_scores(scores, logits_dataset['overflow_to_sample_mapping'], func_name='mean')

    results.append(
        [f"{model_name}_{template[0]}", dataset_name] + 
        measure_classification(dataset['target'], scores, multilabel=multilabel) + 
        [round(end_time - start_time, 8)]
    )

    return results


def decode_batch(batch:dict, model) -> dict:
    """Get the loss and perplexity of text generation on a batch.

    Args:
        batch (dict): input_ids and attention_mask of the text batch to process.
        model: model to use for text generation.

    Returns:
        dict: loss and perplexity values of the batch.
    """
    model_input = {
        'input_ids': torch.tensor(batch['input_ids']).to(device=device),
        'attention_mask': torch.tensor(batch['attention_mask']).to(device=device)
    }
    with torch.no_grad():
        loss, scores = model(model_input)

    return {'loss': loss, 'perplexity': scores}


def decode(dataset, labels:list, model, tokenizer, model_name:str, batch_size:list, 
    tokenizer_kwargs:dict, dataset_name:str, start_indexes:list=[-1]) -> list:
    """Perform text generation of a dataset.

    Args:
        dataset: dataset to process.
        labels (list): name of the classes/labels.
        model: model to perform the task.
        tokenizer: tokenizer of the model.
        model_name (str): name of the model.
        batch_size (list): list with batch sizes for tokenizer, model and general dataset processing.
        tokenizer_kwargs (dict): arguments to be passed to the tokenizer.
        dataset_name (str): name of the dataset.
        start_indexes (list, optional): index from which perplexity is calculated. Defaults to [-1].

    Returns:
        list: text generation results.
    """
    # Variables
    results:list = []
    file_name = f"{file_path}temp/decode_{dataset_name}_{model_name}"
    tokenizerGen = GenTokenizer(tokenizer, tokenizer_kwargs, start_indexes)
    decoder = Decoder(model, start_indexes).to(device=device)

    # Processing
    start_time = time()
    decoded_dataset = dataset.map(lambda x: tokenizerGen.tokenize(x['text']), 
        batched=True, batch_size=batch_size[0], remove_columns=dataset.column_names)
    decoded_dataset = decoded_dataset.map(lambda x: decode_batch(x, decoder), batched=True, 
        batch_size=batch_size[-1])
    decoded_dataset.save_to_disk(file_name)
    end_time = time()

    scores = torch.tensor(decoded_dataset['perplexity']).numpy()
    for pos, idx in enumerate(start_indexes):
        results.append(
            [f"{model_name}_{idx}", dataset_name] + 
            calculate_statistics(scores[:, pos]) + 
            [round(end_time - start_time, 8)]
        )

    return results


def complete_batch(batch:dict, model, max_new_tokens:int, constr_idx:Optional[torch.tensor]=None) -> dict:
    """Generate text based on a batch.

    Args:
        batch (dict): input_ids and attention_mask of the text batch to process.
        model: model to use for text generation.
        max_new_tokens (int): maximum number of tokens to generate.
        constr_idx (Optional[torch.tensor], optional): mapping of the contraint to the logit space; 
            indices. Defaults to None.

    Returns:
        dict: batch results.
    """
    inputs = {
        'input_ids': torch.tensor(batch['input_ids']).to(device=device),
        'attention_mask': torch.tensor(batch['attention_mask']).to(device=device)
    }
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, output_scores=True, 
        return_dict_in_generate=True)

    to_return = {'output': outputs['sequences'].detach().cpu().numpy()[:,-max_new_tokens:]}
    if not(constr_idx is None):
        logits_constr = outputs['scores'][0][:, constr_idx].detach().cpu().numpy()
        to_return['output_c'] = constr_idx[logits_constr.argmax(axis=1)].reshape(-1,1)
        to_return['logits_c'] = logits_constr

    return to_return


def complete(dataset, labels:list, model, tokenizer, model_name:str, batch_size:list, 
    tokenizer_kwargs:dict, constraint_gen:list, max_new_tokens:int, dataset_name:str, 
    mca:Optional[str]=None, mca_prefix:Optional[str]=None, template:tuple=(None, None)) -> dict:
    """Perform a desired task via text generation, such as text classification of a dataset via 
    multiple-choice question answering.

    Args:
        dataset: dataset to process.
        labels (list): name of the classes/labels.
        model: model to perform the task.
        tokenizer: tokenizer of the model.
        model_name (str): name of the model.
        batch_size (list): list with batch sizes for tokenizer, model and general dataset processing.
        tokenizer_kwargs (dict): arguments to be passed to the tokenizer.
        constraint_gen (list): whether the process is constrained and the constraint type.
        max_new_tokens (int): maximum number of tokens to generate.
        dataset_name (str): name of the dataset.
        mca (Optional[str], optional): string containing the constraint. Defaults to None.
        mca_prefix (Optional[str], optional): string with the constraint's prefix. Defaults to None.
        template (tuple, optional): prompt to be applied, preceded by its id. Defaults to (None, None).

    Returns:
        dict: desired task results.
    """
    # Variables
    results:list = []
    file_name = f"{file_path}temp/complete_{dataset_name}_{model_name}_{template[0]}"
    tokenizer.padding_side = 'left'
    tokenizerGen = MCATokenizer(tokenizer, tokenizer_kwargs)
    model.to(device=device)

    # Constraint
    constraint_flag, constraint = constraint_gen
    if constraint_flag:
        optioning = eval(mca)[:len(labels)] if constraint == 'mca' else constraint
        optioning += [mca_prefix + x for x in optioning]
        constraint_idx = torch.flatten(tokenize_batch(optioning, tokenizer, {'add_special_tokens': False, 
            'return_tensors': 'pt', 'padding': True}).input_ids[:,-max_new_tokens:])
    else:
        constraint_idx = None

    # Processing
    start_time = time()
    tokenizerGen.set_prompt(update_template(labels, template[1], mca))
    completed_dataset = dataset.map(lambda x: tokenizerGen.tokenize(x['feature']), batched=True, 
        batch_size=batch_size[0], remove_columns=dataset.column_names)
    completed_dataset = completed_dataset.map(lambda x: complete_batch(x, model, max_new_tokens, 
        constraint_idx), batched=True, batch_size=batch_size[-1])
    ## Decoding
    completed_dataset = completed_dataset.map(lambda x: {'decoded': detokenize_batch(x['output'], 
        tokenizerGen.tokenizer)}, batched=True, batch_size=batch_size[0])
    end_time = time()

    # Calculating results
    if constraint_flag:
        completed_dataset = completed_dataset.map(lambda x: {'decoded_c': detokenize_batch(
            x['output_c'], tokenizerGen.tokenizer)}, batched=True, batch_size=batch_size[0])
        results.append(
            [f"{model_name}_{template[0]}", dataset_name] + 
            measure_classification(dataset['target'], completed_dataset['logits_c'], True, multilabel) 
            + [round(end_time - start_time, 8)]
        )
    else:
        completed_dataset = completed_dataset.map(lambda x: {'predicted': text2label(x['decoded'], 
            labels, mca)}, batched=True, batch_size=batch_size[0])
        results.append(
            [f"{model_name}_{template[0]}", dataset_name] + 
            measure_classification(dataset['target'], completed_dataset['predicted'], False, multilabel) 
            + [round(end_time - start_time, 8)]
        )
    completed_dataset.save_to_disk(file_name)

    return results


if __name__ == '__main__':
    # Setting-up
    logging.basicConfig(level=logging.DEBUG, filename="logs/logfile.txt", filemode="a+", 
        format="%(asctime)-15s %(levelname)-8s %(message)s")
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        config_path:str = sys.argv[1]
        config_path += '/' if config_path[-1] != '/' else ''
        config:dict = load_json(config_path + 'config.json')
    except:
        config_path:str = './'
        try:
            config:dict = load_json(config_path + 'config.json')
        except:
            print('Exiting. Please provide a path to the config.json file or '
                'locate the file in the same directory as current python file.')
            sys.exit(0)
    data_path:str = config['data']['path']
    file_path:str = config['results']['path']

    # Running inference according to dataset and task
    for dataset_name in config['data']['datasets']:
        # Loading data
        aux = config[dataset_name].get('top_classes', None)
        if not(aux is None):
            r = input(f"Dataset {dataset_name} only with {aux} classes. Press Enter if correct: ")
            if r != '': print('Exiting. Please correct the top_classes argument.'); sys.exit(0)
        dataset, labels = load_data(data_path, **config[dataset_name])
        multilabel = config[dataset_name]['multilabel']

        for task in config['tasks']:
            # Loading templates and assigning task arguments
            templates:dict = load_templates(config['data']['path'] + config[task]['template']['file'], 
                dataset_name)
            task_func = locals()[task]
            task_kwargs:dict = deepcopy(config[task]['kwargs'])
            task_kwargs['dataset_name'] = dataset_name
            use_max_len_model:bool = task_kwargs['tokenizer_kwargs'].get('max_length') in [None, -1]

            for model_ in config[task]['models']:
                # Processing
                results:dict = get_results_template(task)
                model, tokenizer, model_name, max_model_len = get_model_tokenizer(model_, task)
                if use_max_len_model: task_kwargs['tokenizer_kwargs']['max_length'] = max_model_len
                ## W/o template
                if not(config[task]['template']['is_required_flag']):
                    try: results['values'] += task_func(dataset, labels, model, tokenizer, model_name, 
                        **task_kwargs)
                    except AssertionError as msg: logging.info(f'{task} {model} -1 {msg}'); continue
                ## W/ template
                for template in templates.items():
                    try: results['values'] += task_func(dataset, labels, model, tokenizer, model_name, 
                        **task_kwargs, template=template)
                    except AssertionError as msg: logging.info(f'{task} {model_name} {template[0]} {msg}')
                # Saving results
                save_results(results, task, config['results']['path'], config['results']['file'])
