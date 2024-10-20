import torch
import clip
from typing import List

def is_yellow(color, tolerance=0.1):
    return (color[0] > 1 - tolerance and color[1] > 1 - tolerance and color[2] < tolerance)

def load_and_freeze_bert_model(version: str) -> torch.nn.Module:
    """ Load BERT model and freeze its parameters.
    
    Args:
        version: BERT model version.
    
    Return:
        BERT tokenizer and BERT model.
    """
    from transformers import AutoTokenizer, BertModel
    tokenizer = AutoTokenizer.from_pretrained(version, use_fast=False) # don't use fast tokenizer to avoid warning in forking
    bert_model = BertModel.from_pretrained(version)
    
    for p in bert_model.parameters():
        p.requires_grad = False
    
    return tokenizer, bert_model

def encode_text_bert(tokenizer: torch.nn.Module, bert_model: torch.nn.Module, raw_text: List, max_length: int=32, s_feat=False, device: str='cpu') -> torch.Tensor:
    """ Encode text using BERT model.
    
    Args:
        tokenizer: BERT tokenizer.
        bert_model: BERT model.
        raw_text: List of raw text.
        device: Device to use.

    Return:
        If s_feat is True, return detached sentence feature, otherwise return detached language feature of per token and attention mask (1 for real tokens, 0 for padding tokens)
    """
    inputs = tokenizer(raw_text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(device)

    outputs = bert_model(**inputs)
    if s_feat:
        encoded_text = outputs.pooler_output
        return encoded_text.detach()
    else:
        encoded_text = outputs.last_hidden_state
        return encoded_text.detach(), inputs['attention_mask']

def load_and_freeze_clip_model(version: str) -> torch.nn.Module:
    """ Load CLIP model and freeze its parameters.
    
    Args:
        version: CLIP model version.
    
    Return:
        CLIP model.
    """
    clip_model, _ = clip.load(version, device='cpu', jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    return clip_model

def encode_text_clip(clip_model: torch.nn.Module, raw_text: List, max_length: int=32, device: str='cpu') -> torch.Tensor:
    """ Encode text using CLIP model.
    
    Args:
        clip_model: CLIP model.
        raw_text: List of raw text.
        device: Device to use.
    
    Return:
        Detached encoded text.
    """
    if max_length is not None:
        default_context_length = 77
        context_length = max_length + 2
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
        zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)
    else:
        texts = clip.tokenize(raw_text, truncate=True).to(device)
        
    encoded_text = clip_model.encode_text(texts) # [bs, clip_dim]
    return encoded_text.detach()