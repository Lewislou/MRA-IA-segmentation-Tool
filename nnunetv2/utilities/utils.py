from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)

    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]

    files = np.unique(files)
    return files

def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[List[str]]:
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(f + "_\d\d\d\d" + file_ending)
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists
