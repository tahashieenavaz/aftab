import torch
from aftab.mixins import AftabBaseMixin
from aftab.common import _make_sure_directory_exists


class AftabStoreModelMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def _save(self, directory: str, filename: str):
        directory_path = _make_sure_directory_exists(directory).strip("/").strip()
        _filename = f"{directory_path}/{filename}.model"
        torch.save(self._network, _filename)
