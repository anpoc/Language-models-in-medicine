import numpy as np
import torch
from torch import nn
import datasets


class Encoder(nn.Module):
    def __init__(self, model, pool:str):
        """Instantiate an encoder model.

        Args:
            model: model.
            pool (str): pooling strategy. One of: 'CLS_pool', 'max_pool', 'avg_pool'.
        """
        super().__init__()
        self.model = model
        self.pool = globals()[pool]


    def forward(self, x:dict) -> torch.FloatTensor:
        """Perform a forward pass.

        Args:
            x (dict): dict containing the following elements:
                - input_ids: torch.Tensor of the token_type ids with shape (N x M)
                - attention_mask: torch.Tensor containing the attention mask of shape (N x M)

        Returns:
            torch.FloatTensor: sentence_embedding for each sample of shape (N x d_hidden).
        """
        output = self.model(**x)
        return self.pool(output.last_hidden_state, x['attention_mask'])


def CLS_pool(hidden_state: torch.FloatTensor, attention_mask: torch.BoolTensor) -> torch.FloatTensor:
    """Return only the hidden state of the [CLS] token.

    Args:
        hidden_state (torch.FloatTensor): (N x M x d_hidden) where N is the batch size and M is 
            the number of tokens.
        attention_mask (torch.BoolTensor): (N x M), True for "real" tokens, False for padding tokens.

    Returns:
        torch.FloatTensor: (N x d_hidden).
    """
    return hidden_state[:,0,:]


def max_pool(hidden_state: torch.FloatTensor, attention_mask: torch.BoolTensor) -> torch.FloatTensor:
    """Globally pool the hidden states over all tokens using max pooling.

    Args:
        hidden_state (torch.FloatTensor): (N x M x d_hidden) where N is the batch size and M is 
            the number of tokens.
        attention_mask (torch.BoolTensor): (N x M), True for "real" tokens, False for padding tokens.

    Returns:
        torch.FloatTensor: (N x d_hidden).
    """
    hidden_state[attention_mask == 0] = float('-inf')
    return torch.max(hidden_state, dim=1).values


def avg_pool(hidden_state: torch.FloatTensor, attention_mask: torch.BoolTensor) -> torch.FloatTensor:
    """Globally pool the hidden states over all tokens using average pooling.

    Args:
        hidden_state (torch.FloatTensor): (N x M x d_hidden) where N is the batch size and M is 
            the number of tokens.
        attention_mask (torch.BoolTensor): (N x M), True for "real" tokens, False for padding tokens.

    Returns:
        torch.FloatTensor: (N x d_hidden).
    """
    filtered_values = torch.mul(hidden_state, attention_mask[:,:,None])
    noPadding_size = attention_mask.sum(dim=1)[:, None]
    return torch.div(filtered_values.sum(dim=1), noPadding_size)


def aggregate_update_embedding(pool: str, ds_original: datasets.Dataset, 
    ds_embedding: datasets.Dataset, col_name: str) -> datasets.Dataset:
    """Aggregate several embeddings representing one sample into one embedding and update the dataset.

    Args:
        pool (str): pooling strategy.
        ds_original (datasets.Dataset): raw dataset.
        ds_embedding (datasets.Dataset): embedding dataset.
        col_name (str): name of the column that contains the embeddings.

    Returns:
        datasets.Dataset: aggregated and updated dataset.
    """
    if ds_embedding.num_rows != ds_original.num_rows:
        df = ds_embedding.to_pandas()[['attention_mask', 'overflow_to_sample_mapping', col_name]]\
            .groupby('overflow_to_sample_mapping')
        if pool == 'CLS_pool':
            embedding = df[col_name].agg(lambda x: np.vstack(x).mean(axis=0)).values.tolist()
        elif pool == 'max_pool':
            embedding = df[col_name].agg(lambda x: np.vstack(x).max(axis=0)).values.tolist()
        else:
            embedding = df.apply(lambda x: np.average(np.vstack(x[col_name]), axis=0, 
                weights=np.vstack(x['attention_mask']).sum(axis=1))).values.tolist()
    else: 
        embedding = ds_embedding[col_name]
    ds_original = ds_original.add_column(col_name, embedding)

    return ds_original


def cosine_similarity(x: torch.FloatTensor, y: torch.FloatTensor) -> np.ndarray:
    """Calculate cosine similarity between two tensors.

    Args:
        x (torch.FloatTensor): tensor.
        y (torch.FloatTensor): tensor.

    Returns:
        np.ndarray: cosine similarity between x and y.
    """
    distance = nn.functional.normalize(x) @ nn.functional.normalize(y).t()
    return distance.cpu().numpy()


def softmax(logits: torch.FloatTensor, dim=1) -> torch.FloatTensor:
    """Apply softmax on a tensor.

    Args:
        logits (torch.FloatTensor): logit tensor.
        dim (int, optional): dimension along which to apply the softmax. Defaults to 1.

    Returns:
        torch.FloatTensor: softmax tensor.
    """
    return nn.functional.softmax(logits, dim=dim)