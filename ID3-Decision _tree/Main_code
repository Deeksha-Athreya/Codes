import torch
import math

def get_entropy_of_dataset(t: torch.Tensor):
    """Calculate the entropy of the entire dataset"""
    target_col = t[:, -1] 
    uniq_classes, counts = torch.unique(target_col, return_counts=True)
    
    total = target_col.size(0)
    
    ent = 0.0
    for c in counts:
        prob = c.item() / total
        ent -= prob * math.log2(prob)
    
    return ent

def get_avg_info_of_attribute(t: torch.Tensor, attr: int):
    """Return avg_info of the attribute provided as parameter"""
    attr_col = t[:, attr]
    uniq_vals, counts = torch.unique(attr_col, return_counts=True)
    
    total = t.size(0)
    
    avg_info = 0.0
    for val, c in zip(uniq_vals, counts):
        subset = t[attr_col == val]
        ent = get_entropy_of_dataset(subset)
        prob = c.item() / total
        avg_info += prob * ent
    
    return avg_info

def get_information_gain(t: torch.Tensor, attr: int):
    """Return Information Gain of the attribute provided as parameter"""
    ent_dataset = get_entropy_of_dataset(t)
    avg_info = get_avg_info_of_attribute(t, attr)
    info_gain = ent_dataset - avg_info
    
    return info_gain

def get_selected_attribute(t: torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute
    """
    num_attrs = t.size(1) - 1  
    ig_dict = {}
    
    for attr in range(num_attrs):
        ig = get_information_gain(t, attr)
        ig_dict[attr] = ig
    
    sel_attr = max(ig_dict, key=ig_dict.get)
    
    return ig_dict, sel_attr
