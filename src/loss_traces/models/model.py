import os
from typing import Tuple

import torch
from torch.nn import Module
from torchvision.models import resnet18, vgg11

from loss_traces.config import MODEL_DIR
from loss_traces.models.simple_convnet import Net


class ModelLoader:
    @classmethod
    def _load_simple_convnet(cls, num_classes: int) -> Net:
        from loss_traces.models.simple_convnet import Net
        model = Net()
        return model

    @classmethod
    def _load_resnet18(cls, num_classes: int) -> Tuple[Module, Module]:
        model = resnet18(num_classes=num_classes)
        return model

    @classmethod
    def _load_vgg11(cls, num_classes: int) -> Tuple[Module, Module]:
        model = vgg11(num_classes=num_classes)
        return model

    @classmethod
    def _load_resnet20(cls, num_classes: int) -> Tuple[Module, Module]:
        from loss_traces.models.resnet_cifar import resnet20
        model = resnet20(num_classes=num_classes)
        return model

    @classmethod
    def _load_wide_resnet(cls, num_classes: int) -> Module:
        from loss_traces.models.wide_resnet import WideResNet
        model = WideResNet(28, num_classes, 2)
        return model

    @classmethod
    def _load_even_wide_resnet_28_10(cls, num_classes: int) -> Module:
        from loss_traces.models.wide_resnet import WideResNet
        model = WideResNet(28, num_classes, 10)
        return model

    @classmethod
    def _load_wide_resnet_40_2(cls, num_classes: int) -> Module:
        from loss_traces.models.wide_resnet import WideResNet
        model = WideResNet(40, num_classes, 2)
        return model

    @classmethod
    def _load_wide_resnet_40_4(cls, num_classes: int) -> Module:
        from loss_traces.models.wide_resnet import WideResNet
        model = WideResNet(40, num_classes, 4)
        return model

    @classmethod
    def _load_vgg16(cls, num_classes: int) -> Module:
        from torchvision.models import vgg16
        model = vgg16(num_classes=num_classes)
        return model

    @classmethod
    def _load_densenet121(cls, num_classes: int) -> Module:
        from torchvision.models import densenet121
        model = densenet121(num_classes=num_classes)
        return model

    @classmethod
    def _load_mobilenet_v2(cls, num_classes: int) -> Module:
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(num_classes=num_classes)
        return model

    # Mapping of architecture names to their corresponding loader methods
    _ARCH_LOADERS = {
        'simple_convnet': _load_simple_convnet,
        'vgg11': _load_vgg11,
        'rn-20': _load_resnet20,
        'rn-18': _load_resnet18,
        'wrn28-2': _load_wide_resnet,
        'wrn28-10': _load_even_wide_resnet_28_10,
        'wrn40-2': _load_wide_resnet_40_2,
        'wrn40-4': _load_wide_resnet_40_4,
        'vgg16': _load_vgg16,
        'densenet121': _load_densenet121,
        'mobilenetv2': _load_mobilenet_v2
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