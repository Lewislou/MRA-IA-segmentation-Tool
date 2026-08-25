from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np

class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):

        for i in input_list[1:]:
            if not len(i) == len(input_list[0]):
                return False
            all_same = all(i[j] == input_list[0][j] for j in range(len(i)))
            if not all_same:
                return False
        return True

    @staticmethod
    def _check_all_same_array(input_list):

        for i in input_list[1:]:
            if not all([a == b for a, b in zip(i.shape, input_list[0].shape)]):
                return False
            all_same = np.all(np.isclose(i, input_list[0]))
            if not all_same:
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        pass
