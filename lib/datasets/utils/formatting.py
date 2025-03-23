from collections.abc import Sequence
import mmcv
import numpy as np
import torch


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class ToTensor(object):
    """Convert some sample to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        """Call function to convert data in sample to :obj:`torch.Tensor`.

        Args:
            sample (Sample): sample data contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """

        for key in self.keys:
            if isinstance(sample[key], list):
                for i in range(len(sample[key])):
                    sample[key][i] = to_tensor(sample[key][i])
            else:
                sample[key] = to_tensor(sample[key])
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'



