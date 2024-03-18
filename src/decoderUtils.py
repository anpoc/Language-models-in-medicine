import numpy as np
import torch
from torch import nn


class Decoder(nn.Module):
    """This module is based on the following sources:
    https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/ac4135177bfee71b1efd7bd3aff62e456e30aef9/perplexity.py
    https://huggingface.co/docs/transformers/perplexity
    """
    def __init__(self, model, start_indexes:list):
        """Instantiate a decoder model.

        Args:
            model: model.
            start_indexes (list): indices from which the perplexity calculation starts.
        """
        super().__init__()
        self.model = model
        self.start_idx = np.array(start_indexes) + 1 * (-1 in start_indexes)


    def forward(self, x:dict):
        """Perform a forward pass.

        Args:
            x (dict): dict containing the following elements:
                - input_ids: torch.Tensor of the token_type ids with shape (N x M)
                - attention_mask: torch.Tensor containing the attention mask of shape (N x M)
                - position_ids: torch.Tensor containing the position ids of shape (N x M)

        Returns:
            losses and perplexities.
        """
        losses = []
        scores = []

        for offset in self.start_idx:
            x_ = shift_inputs(x, offset)
            output = self.model(**x_)

            shift_logits = output['logits'][..., :-1, :].contiguous()
            shift_labels = x_['input_ids'][..., 1:].contiguous()
            shift_att_mask = x_['attention_mask'][..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.transpose(1,2), shift_labels, 
                reduction='none')
            score = torch.exp2((loss * shift_att_mask).sum(dim=1) / shift_att_mask.sum(dim=1))
            losses.append(loss.detach().cpu().numpy())
            scores.append(score.detach().cpu().numpy())

        return transform_array_to_list(losses), np.array(scores).T


def shift_inputs(x:torch.FloatTensor, offset:int) -> torch.FloatTensor:
    """Shift a tensor by a given offset.

    Args:
        x (torch.FloatTensor): tensor.
        offset (int): offset.

    Returns:
        torch.FloatTensor: shifted tensor.
    """    
    x_shifted = {}
    for key, val in x.items():
        x_shifted[key] = val[..., offset:]
    return x_shifted


def transform_array_to_list(x: np.ndarray) -> list:
    """Transform an array into a list of rows (np.array).

    Args:
        x (np.ndarray): array.

    Returns:
        list: list of rows of the given array.
    """
    return [[xn[i,:] for xn in x] for i in range(x[0].shape[0])]
