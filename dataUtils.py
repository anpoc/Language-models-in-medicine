import pandas as pd
import numpy as np
import json, os, pickle, re

from typing import Optional
from datasets import Dataset, load_from_disk
from copy import deepcopy
from tqdm import tqdm

from mimiccxrUtils import *


global dataset_df


def apply_template(context:pd.DataFrame, template:dict) -> list:
    """Apply a template/prompt over some text.

    Args:
        context (pd.DataFrame): texts to apply the template over.
        template (dict): template/prompt.

    Returns:
        list: modified texts.
    """
    return list(map(lambda x: template['text'].format(report=x), context))


def update_template(labels:list, template:dict, mca:Optional[str]=None) -> str:
    """Update a template given some labels and possible additional encoding options of labels 
        as choices.

    Args:
        labels (list): labels.
        template (dict): template/prompt.
        mca (Optional[str], optional): additional encoding options of labels.

    Returns:
        str: _description_
    """
    other_requirements = {}
    if 'options' in template['requirements']:
        optioning = eval(mca)
        other_requirements['options'] = template['options_settings']['separator'].join(
            ['(' + optioning[idx] + ') ' + label for idx, label in enumerate(labels)])
    return template['text'].format(report='{report}', **other_requirements)


def label_binarization(n_labels: int) -> None:
    """One hot encoding of labels.

    Args:
        n_labels (int): number of labels.
    """
    global dataset_df
    binarization = lambda x, n: np.isin(np.arange(n), x).astype(int)
    temp = dataset_df.groupby('feature').target.apply(lambda x: binarization(list(x.values), n_labels))
    dataset_df = dataset_df[['feature']].merge(temp.reset_index(), on='feature', how='left')


def check_existence_load(file_path:str, constraint:int=None):
    """Checking whether the the file exists or not given a contraint, and return it if it exists 
        and follows the constraint.

    Args:
        file_path (str): path where the file is stored.
        constraint (int, optional): indicating a contraint in terms of number of rows the file 
            should contain. Defaults to None.

    Returns:
        whether the file exists and follows the contraint and the file.
    """
    pending:bool = True
    file_ = None

    if os.path.exists(file_path):
        file_ = load_from_disk(file_path)
        pending = False
        if not(constraint is None) and ('overflow_to_sample_mapping' in file_.features):
            if not(constraint in file_[-1].values()): file_ = None; pending = True
        elif not(constraint is None) and (constraint != (file_.num_rows - 1)): file_ = None; pending = True

    return pending, file_


def load_data(folder_path:str, file_name:str, feature_col:Optional[str]=None, 
    target_col:Optional[str]=None, top_classes:Optional[int]=None, multilabel:bool=False):
    """Load any of the datasets. This is the function which should be called.

    Args:
        folder_path (str): path to the folder where the data is stored.
        file_name (str): name of the file containing the dataset.
        feature_col (Optional[str], optional): name of the column containing the features 
            (i.e. texts or reports). Defaults to None.
        target_col (Optional[str], optional): name of the column containing the labels. 
            Defaults to None.
        top_classes (Optional[int], optional): number of classes to be considered according to 
            frequency (descending). Defaults to None.
        multilabel (bool, optional): whether the dataset is multilabel or not (i.e. multiclass). 
            Defaults to False.

    Returns:
        dataset and list of labels
    """
    global dataset_df

    if 'mimic_cxr' in file_name:
        # Loading data
        #dataset_df = load_mimic_cxr_dataset(folder_path + file_name, extension='.txt')
        #dataset_df.to_csv(folder_path + 'mimic_cxr.csv')
        #dataset_df = dataset_df.drop_duplicates(subset='text')
        dataset_df = pd.read_csv(folder_path + 'mimic_cxr.csv').dropna()
        dataset_df = dataset_df.loc[dataset_df['text'] != ''][:10].reset_index(drop=True)
        dataset_df['text'] = dataset_df['text'].apply(lambda x: re.sub('\s+', ' ', 
            re.sub('^\s+|\\n|\s+$', '', x)))
        # Defining labels
        labels = None
    else:
        assert((feature_col is not None) and (target_col is not None)), \
            'Please specify both feature and target columns on the config.json file.'
        # Loading data
        dataset_df = load_csv_dataset(folder_path + file_name, feature_col, target_col)
        # Preprocessing datasets
        if 'transcriptions' in file_name: preprocess_transcriptions()
        # Restricting to top classes
        if not(top_classes is None):
            top_values = dataset_df['target'].value_counts().sort_values(ascending=False)\
                .head(top_classes).index.sort_values().tolist()
            dataset_df.loc[~dataset_df['target'].isin(top_values), 'target'] = 'Other'
        # Defining labels
        dataset_df['target'] = dataset_df['target'].astype('category')
        if not(top_classes is None): 
            dataset_df['target'] = dataset_df['target'].cat.reorder_categories(top_values + ['Other'])
        labels = dataset_df['target'].cat.categories.tolist()
        dataset_df['target'] = dataset_df['target'].cat.codes
        if multilabel: label_binarization(n_labels=len(labels))
        dataset_df.drop_duplicates(subset=['feature'], inplace=True)
    # Converting into dataset
    dataset = Dataset.from_pandas(dataset_df, preserve_index=False)

    return dataset, labels


def load_csv_dataset(file_path:str, feature_col:str, target_col:str) -> pd.DataFrame:
    """Load a dataset in .csv format.

    Args:
        file_path (str): path where the .csv file is stored.
        feature_col (str): name of the column containing the features (i.e. texts or reports).
        target_col (str): name of the column containing the labels.

    Returns:
        pd.DataFrame: dataset.
    """
    # Loading dataset, renaming columns, and dropping NAs
    df = pd.read_csv(file_path, usecols=[feature_col, target_col])
    df.rename(columns={feature_col: 'feature', target_col: 'target'}, inplace=True)
    df.dropna(inplace=True)

    # Cleaning target column
    df['target'] = df['target'].str.strip()

    return df


def load_mimic_cxr_dataset(files_path:str, extension:str='.txt') -> pd.DataFrame:
    """Load MIMIC-CXR dataset. This function based on 
        https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv-cxr/txt/

    Args:
        files_path (str): path where the files are stored.
        extension (str, optional): files type. Defaults to '.txt'.

    Returns:
        pd.DataFrame: MIMIC-CXR dataset.
    """
    # Setting-up
    remove_spacing = lambda x: re.sub('\s+', ' ', re.sub('^\s+|\\n|\s+$', '', x))
    dataset = {
        'file_name': [],
        'section': [],
        'text': []
    }
    custom_section_names, custom_indices = get_custom_rules()

    # Getting the files to read
    files_list = []
    for root, _, files in os.walk(files_path):
        files_list += [os.path.join(root, f) for f in files if f.endswith(extension)]

    # Reading and processing files
    for file_path in tqdm(files_list):
        with open(file_path, 'r') as f:
            content = ''.join(f.readlines())
        file_name = '/'.join(file_path[:-len(extension)].split('/')[-3:])

        # Checking whether file is in the list of special files
        file_subname = file_name.split('/')[-1]
        if file_subname in custom_indices:
            idx = custom_indices[file_subname]
            dataset['file_name'].append(file_name)
            dataset['section'].append('special')
            dataset['text'].append(remove_spacing(content[idx[0]:idx[1]]))
            continue

        # Extracting the sections
        sections, section_names, section_idx = section_text(content, r'\n ([A-Z ()/,-]+):\s')
        section_names = normalize_section_names(section_names)

        # Checking for misnamed sections
        if file_subname in custom_section_names:
            sn = custom_section_names[file_subname]
            idx = list_rindex(section_names, sn)
            dataset['file_name'].append(file_name)
            dataset['section'].append('misnamed')
            dataset['text'].append(sections[idx])
            continue

        # Getting last section
        idx = -1
        for sn in ('impression', 'findings', 'last_paragraph', 'comparison'):
            if sn in section_names:
                idx = list_rindex(section_names, sn)
                break
        if idx != -1:
            dataset['file_name'].append(file_name)
            dataset['section'].append(sn)
            dataset['text'].append(sections[idx])

    dataset = pd.DataFrame.from_dict(dataset).dropna()
    dataset = dataset.loc[dataset['text'] != '']

    return dataset


def load_templates(file_path:str, dataset_name:str) -> dict:
    """Load the templates.

    Args:
        file_path (str): path where the templates are stored.
        dataset_name (str): target dataset.

    Returns:
        dict: templates.
    """    
    # Uploading as csv file
    if file_path.endswith('.csv'):
        templates = pd.read_csv(file_path, sep='|').fillna('')
        templates = templates[(templates['status']) & (templates['data'].str.contains(dataset_name))]
        return templates.set_index('ID').T.to_dict('dict')
    # Uploading as json file
    elif file_path.endswith('.json'):
        file_ = load_json(file_path)
        templates = {}
        for temp_k in file_[dataset_name]['templates']:
            temp_v = file_['templates'][temp_k]
            # If question is not required, save and continue
            if not('question' in temp_v['requirements'] or 'QUESTION' in temp_v['requirements']):
                templates[temp_k] = deepcopy(temp_v)
                continue
            # If question is required, add it and save
            for ques_k in file_[dataset_name]['questions']:
                ques_v = file_['questions'][ques_k].lower() if 'question' in temp_v['requirements'] \
                    else file_['questions'][ques_k]
                temp_v_ = deepcopy(temp_v)
                temp_v_['text'] = re.sub('{question}', ques_v, temp_v['text'])
                templates[temp_k + '_' + ques_k] = temp_v_
        return templates
    else:
        return {}


def load_json(file_path:str) -> dict:
    """Load a json file.

    Args:
        file_path (str): path where the json is stored.

    Returns:
        dict: json.
    """    
    with open(file_path, 'rb') as f:
        item = json.load(f)
    return item


def load_pickle(file_path:str):
    """Load a pickle file.

    Args:
        file_path (str): path where the pickle is stored.

    Returns:
        item that the pickle stored.
    """    
    with open(file_path, 'rb') as f:
        item = pickle.load(f)
    return item


def preprocess_transcriptions():
    """Preporcess transcriptions dataset: renaming and retaining important labels and modifying 
        reports format.
    """
    global dataset_df

    # Proposed labels
    labels_norename = [
        'Autopsy', 'Bariatrics', 'Dentistry', 'Dermatology', 'Diets and Nutritions', 
        'Endocrinology', 'Gastroenterology', 'Nephrology', 'Neurology', 'Ophthalmology', 
        'Radiology', 'Sleep Medicine', 'Urology', 'Orthopedic', 'Neurosurgery', 'Podiatry', 
        'Rheumatology', 'Surgery'
    ]
    labels_rename = {
        'Allergy / Immunology': 'Allergy or Immunology',
        'Cardiovascular / Pulmonary': 'Cardiovascular or Pulmonary',
        'Chiropractic': 'Physical Medicine and Rehabilitation, or Chiropractic',
        'Cosmetic / Plastic Surgery': 'Cosmetic or Plastic Surgery',
        'ENT - Otolaryngology': 'Otolaryngology',
        'Hematology - Oncology': 'Hematology or Oncology',
        'Lab Medicine - Pathology': 'Laboratory Medicine or Clinical Pathology',
        'Obstetrics / Gynecology': 'Obstetrics or Gynecology',
        'Pediatrics - Neonatal': 'Pediatrics or Neonatal',
        'Physical Medicine - Rehab': 'Physical Medicine and Rehabilitation, or Chiropractic',
        'Psychiatry / Psychology': 'Psychiatry or Psychology',
        'Speech - Language': 'Speech and Language'
    }
    # Preprocessing targets: masking and renaming
    dataset_df = dataset_df.loc[dataset_df['target'].isin(labels_norename + list(labels_rename.keys()))]
    dataset_df.loc[:, 'target'] = dataset_df.loc[:, 'target'].replace(labels_rename).values.tolist()
    # Preporcessing features: cleaning
    dataset_df.loc[:, 'feature'] = dataset_df.loc[:, 'feature'].str.replace(' +', ' ', regex=True)\
        .str.replace(',(?=[A-z ]+:\s*,)', '\n\n', regex=True)\
        .str.replace(':\s*,\s*', ':\n', regex=True)\
        .str.replace('(?=.\s*),(?=[A-z0-9])' , '\n', regex=True).values.tolist()


def text2label(context:list, labels:list, mca:str) -> list:
    """Convert choices (text) into corresponding labels.

    Args:
        context (list): choices.
        labels (list): labels.
        mca (str): additional encoding options of labels as choices.

    Returns:
        list: choices converted into labels.
    """    
    replace = dict(zip(eval(mca)[:len(labels)], range(len(labels))))
    return list(map(lambda x: replace[x] if x in replace.keys() else -1, context))


def save_results(results:dict, task:str, folder_path:str, file_name:str):
    """Save the results.
    
    Args:
        results (dict): results to be stored.
        task (str): approach or task from which the results come from.
        folder_path (str): path where the results are stored.
        file_name (str): name under which the results are stored.
    """
    df_results = pd.DataFrame(results['values'], columns=results['col_names'])
    df_results['task'] = task
    file_path = folder_path + f'results_{task}.csv' if file_name is None else file_name
    header_flag = not(os.path.isfile(file_path))
    df_results.to_csv(file_path, mode='a', header=header_flag, index=False, sep='|')


def save_pickle(item, file_path:str):
    """Save an item as pickle.

    Args:
        item: item to be saved.
        file_path (str): path where the item is stored.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(item, f)