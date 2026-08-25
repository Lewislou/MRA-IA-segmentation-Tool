import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers

class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__()

        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, "%s.npz" % c)
            self.dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, "%s.npz" % c)

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and\
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']

        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data

        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']

        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg

        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']

if __name__ == '__main__':

    folder = '/path/to/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)

    ks = ds['liver_0'].keys()
    assert 'properties' in ks

    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=1000)

    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))

    ks = ds['liver_0'].keys()
    assert 'properties' in ks

    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)

    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))

    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')

        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))
