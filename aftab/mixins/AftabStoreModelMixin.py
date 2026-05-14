import torch
from aftab.mixins import AftabBaseMixin
from aftab.common import _make_sure_directory_exists


class AftabStoreModelMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def _save(self, directory: str):
        directory_path = _make_sure_directory_exists(directory).strip("/").strip()
        filename = f"{directory_path}/{self.__make_network_filename()}.model"
        torch.save(self._network, filename)
