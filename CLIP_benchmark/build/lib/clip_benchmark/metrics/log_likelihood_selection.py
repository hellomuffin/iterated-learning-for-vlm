import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn



def batched_sentence_log_likelihood(logits, labels, pad_id=0):
    """
    Compute the log likelihood of sentences in a batch given model logits and actual labels,
    ignoring contributions from padding tokens.

    :param logits: Tensor of shape (batch_size, sequence_length, vocab_size) - Logits from the model.
    :param labels: Tensor of shape (batch_size, sequence_length) - Actual labels (word indices in the vocabulary).
    :param pad_id: int - The index of the padding token in the vocabulary to be ignored in the loss calculation.
    :return: Tensor representing the log likelihood of each sentence in the batch, ignoring padding.
    """
    # Convert logits to log probabilities
    log_probs = F.log_softmax(logits, dim=2)
    
    # Create a mask to ignore padding
    mask = (labels != pad_id).float()
    
    # Flatten the tensors to align all sequences
    log_probs = log_probs.view(-1, log_probs.size(-1))
    labels = labels.view(-1)
    mask = mask.view(-1)
    
    # Gather the log probabilities of the actual words across all sequences
    actual_word_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
    
    # Apply mask to ignore padding
    actual_word_log_probs = actual_word_log_probs * mask
    
    # Reshape back to (batch_size, sequence_length) to compute sum over sequence_length for each batch
    actual_word_log_probs = actual_word_log_probs.view(logits.size(0), logits.size(1))
    
    # Sum the log probabilities for each sentence in the batch to get the log likelihood, ignoring padding
    sentence_log_likelihoods = actual_word_log_probs.sum(dim=1) / mask.view(logits.size(0), logits.size(1)).sum(dim=1)
    
    return sentence_log_likelihoods




def evaluate(model, dataloader, tokenizer,  device, amp=True):
    """
    Evaluate the model on the given dataset.
    The task has N instances, each instance has I images and C captions.
    For each instance, the goal is to find the correct image for each caption and the correct caption for each image.
    This is done by computing the similarities between each image and each caption.
    This procedure is used to evaluate the models on Winoground and SugarCrepe.

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision
    
    Returns
    -------
    
    dict of accuracy metrics
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    image_score = []
    score = []
    model = model.to(torch.float64)
    with torch.no_grad(), autocast():
        for batch_images, batch_texts in tqdm(dataloader):
            logDict = dict()
            for i, tag in enumerate(['positive', 'negative']):
                texts_tok = tokenizer([t[i] for t in batch_texts]).to(device)
                batch_images = batch_images.to(device).to(torch.float64)
                model_out = model(batch_images, texts_tok)
                logits, labels = model_out["logits"], model_out["labels"]    
                # logllh = batched_sentence_log_likelihood(logits, labels)
                logllh = torch.mean(-criterion(logits.permute(0,2,1), labels), dim=-1)
                logDict[tag] = logllh
            scores = (logDict['positive'] >= logDict['negative']).int()
            image_score.append(scores)
    
    score = torch.cat(image_score, dim=0)

    metrics = {}
    metrics["acc"] = torch.Tensor(score).float().mean().item()
    print("acc", metrics["acc"])
    return metrics