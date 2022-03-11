# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    encoding_dict = {
        'A' : [1, 0, 0, 0],
        'T' : [0, 1, 0, 0],
        'C' : [0, 0, 1, 0],
        'G' : [0, 0, 0, 1]
    }
    
    encoding_list = []
    for sl in seq_arr:
        encoding_list_tmp = []
        for s in sl:
            encoding_list_tmp = encoding_list_tmp + encoding_dict[s] 
        encoding_list.append(encoding_list_tmp)
    return encoding_list


def sample_seqs(seqs,labels, num_out_seqs='equal'):
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    
    pos_seq = [seqs[i] for i in np.where(labels == 1)[0]]
    neg_seq = [seqs[i] for i in np.where(labels == 0)[0]]
    if num_out_seqs == 'equal':
        num_out_seqs = len(pos_seq)

    seq_lenth = len(pos_seq[0])
    pos_seq = set(pos_seq)
    
    
    out_seq=[]
    while len(out_seq) < num_out_seqs:
        # pick a random seq of the neg sequences
        rand_idx = np.random.randint(0, len(neg_seq))
        rand_start_read = np.random.randint(0, len(neg_seq[rand_idx])- seq_lenth)
        out_seq_tmp = neg_seq[rand_idx][rand_start_read:rand_start_read+seq_lenth]
        if out_seq_tmp not in pos_seq: # check to make sure it's not a positive
            if out_seq_tmp not in out_seq: # check to make sure we don't already have it
                out_seq.append(out_seq_tmp)

    out_seq_all = np.append(np.array(list(pos_seq)), out_seq)
    labels_all = np.append(np.ones(len(pos_seq)),np.zeros(len(out_seq)))

    return out_seq_all, labels_all

