from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
import json
import os
import torch


def load_scam_dataset(data_path, output_dir=None):
    ds = load_dataset(data_path)
    ds.save_to_disk(output_dir)

def load_from_local(data_path):
    ds = load_from_disk(data_path)
    return ds

if __name__ == "__main__":
    data_path = "BothBosu/youtube-scam-conversations"
    output_dir = "ai-in-the-loop/data/ytsc_dataset"
    load_scam_dataset(data_path, output_dir)
    # print(load_from_local(output_dir)['train'][0])
