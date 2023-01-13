import pytest
import torch

from muzero.nn import (
    ShortcutProjection,
    ResidualBlock,
    BottleneckResidualBlock,
    RepresentationNetwork,
)

def test_residual_block():
    in_channels = 3
    out_channels = 5
    stride = 1

    n_images = 10
    img_height = 128
    img_width = 128

    block = ResidualBlock(in_channels, out_channels, stride)
    x = torch.randn(n_images, in_channels, 128, 128)

    output = block(x)

    assert output.shape == (n_images, out_channels, img_height, img_width)

def test_shortcut_projection():
    in_channels = 3
    out_channels = 5
    stride = 1

    n_images = 10
    img_height = 128
    img_width = 128

    block = ShortcutProjection(in_channels, out_channels, stride)
    x = torch.randn(n_images, in_channels, img_height, img_width)

    output = block(x)

    assert output.shape == (n_images, out_channels, img_height, img_width)


def test_bottleneck_residual_block():
    in_channels = 3
    bottleneck_channels = 5
    out_channels = 10
    stride = 1

    n_images = 10
    img_height = 128
    img_width = 128

    block = BottleneckResidualBlock(in_channels, bottleneck_channels, out_channels, stride)
    x = torch.randn(n_images, in_channels, img_height, img_width)

    output = block(x)

    assert output.shape == (n_images, out_channels, img_height, img_width)

def test_create_representation_network():
    img_channels = 3
    n_blocks = [3, 3, 3]
    n_channels = [16, 32, 64]
    bottlenecks = None  # [8, 16, 16]
    first_kernel_size = 7


    representation = RepresentationNetwork(
        img_channels=img_channels, n_blocks=n_blocks, n_channels=n_channels,
        first_kernel_size=first_kernel_size, bottlenecks=bottlenecks
    )

    assert representation.conv.kernel_size == (7, 7)


def test_create_bottleneck_representation():
    pass