import torch
from workbench.models import *

def test_anatomynet():
    model = AnatomyNet3D(num_classes=3, in_channel=1)
    in_tensor = torch.rand(1, 1, 8, 8, 8)
    ideal_out = torch.rand(1, 3, 8, 8, 8)
    output = model(in_tensor)
    assert output.shape == ideal_out.shape

def test_bdclstm():
    model = BDCLSTM(input_channels=64, num_classes=3)
    in_tensor = torch.rand(32, 64, 8, 8)
    in_tensor2 = torch.rand(32, 64, 8, 8)
    in_tensor3 = torch.rand(32, 64, 8, 8)
    ideal_out = torch.rand(32, 3, 8, 8)
    output = model(in_tensor, in_tensor2, in_tensor3)
    assert output.shape == ideal_out.shape

def test_densevoxel():
    model = DenseVoxelNet(in_channels=1, num_classes=3)
    in_tensor = torch.rand(1, 1, 8, 8, 8)
    ideal_out = torch.rand(1, 3, 8, 8, 8)
    output = model(in_tensor)
    assert output[0].shape == ideal_out.shape
    assert output[1].shape == ideal_out.shape




