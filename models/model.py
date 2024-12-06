import os
from typing import Tuple

import torch
from opacus.validators import ModuleValidator
from torch.nn import Module
from torchvision.models import resnet18, vgg11

from config import MODEL_DIR
from models.simple_convnet import Net


## TODO: Remove irrelevant models + possibly the "part_to_train" convention if we're not using it
class ModelLoader:
    @classmethod
    def _load_simple_convnet(cls, num_classes: int) -> Net:
        from models.simple_convnet import Net
        model = Net()
        return model

    @classmethod
    def _load_resnet18(cls, num_classes: int) -> Tuple[Module, Module]:
        model = resnet18(num_classes=num_classes)
        model = ModuleValidator.fix(model)
        return model

    @classmethod
    def _load_vgg11(cls, num_classes: int) -> Tuple[Module, Module]:
        model = vgg11(num_classes=num_classes)
        return model

    @classmethod
    def _load_resnet20(cls, num_classes: int) -> Tuple[Module, Module]:
        from models.resnet_cifar import resnet20
        model = resnet20()
        return model

    @classmethod
    def _load_wide_resnet(cls, num_classes: int) -> Module:
        from models.wide_resnet import WideResNet
        model = WideResNet(28, num_classes, 2)
        model = ModuleValidator.fix(model)
        return model

    # Mapping of architecture names to their corresponding loader methods
    _ARCH_LOADERS = {
        'simple_convnet': _load_simple_convnet,
        'vgg11': _load_vgg11,
        'rn-20': _load_resnet20,
        'rn-18': _load_resnet18,
        'wrn28-2': _load_wide_resnet,
    }

    @classmethod
    def load_model(cls, arch: str, num_classes: int) -> Module:
        """
        Load a model based on the specified architecture.

        Args:
            arch (str): Name of the model architecture
            num_classes (int): Number of output classes

        Returns:
            Module: The full model

        Raises:
            ValueError: If the specified architecture is not supported
        """
        # Retrieve the loader method for the specified architecture
        loader = cls._ARCH_LOADERS.get(arch)

        if loader is None:
            raise ValueError(f"Architecture '{arch}' is not supported.")

        # Call the loader method with the number of classes
        return loader.__func__(cls, num_classes)


def get_hyperparameter_from_file(exp_id: str, model_id: str) -> dict:
    path = os.path.join(MODEL_DIR, exp_id, model_id)
    saved = torch.load(path)
    print(len(saved['trained_on_indices']))
    try:
        print(saved['arch'])
        print(saved['train_acc'])
        print(saved['test_acc'])
    except:
        pass
    return saved['hyperparameters']


# Example usage
def load_model(arch: str, num_classes: int) -> Module:
    return ModelLoader.load_model(arch, num_classes)