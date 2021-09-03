import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet, model

from .module import ENCODER_ARCH


def add_flow_to_model(base_model):
    # modify the convolution layers
    # Torch models are usually defined in a hierarchical way.
    # nn.modules.children() return all sub modules in a DFS manner
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (5,) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(
        new_kernel_size).contiguous()

    new_conv = nn.Conv2d(5, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]
    # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return base_model


def replace_last_layer(base_model, last_layer_name, out_dim):
    feature_dim = getattr(base_model, last_layer_name).in_features
    setattr(base_model, last_layer_name, nn.Linear(feature_dim, out_dim))
    return base_model


class RGBF_EmbeddingModel(nn.Module):
    """Basic embedding model with single frame features"""

    def __init__(self, model_arch, emb_dim, use_flow, device,
                 pretrained=False):
        super().__init__()
        self.device = device
        self.use_flow = use_flow
        self.emb_dim = emb_dim

        if 'resnet' in model_arch:
            backbone =  ENCODER_ARCH[model_arch].pretrained_init(
                pretrained=pretrained)
            if use_flow:
                backbone = add_flow_to_model(backbone)
            self.resnet = replace_last_layer(backbone, 'fc', emb_dim)
        elif 'effnet' in model_arch:
            effnet_name = 'efficientnet-b{}'.format(model_arch[-1])
            self.effnet = EfficientNet.from_name(
                effnet_name, in_channels=5 if use_flow else 3,
                num_classes=emb_dim, image_size=128)

    def forward(self, x):
        backbone = self.resnet if hasattr(self, 'resnet') else self.effnet
        return backbone(x)

    def embed(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if self.use_flow:
            assert x.shape[1] == 5, 'Wrong number of channels for RGB + flow'
        else:
            assert x.shape[1] == 3, 'Wrong number of channels for RGB'

        self.eval()
        with torch.no_grad():
            return self(x).cpu().numpy()
