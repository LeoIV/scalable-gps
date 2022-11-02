import math
import tqdm
import torch
from torch.nn import Linear, ReLU
from botorch.models.gpytorch import GPyTorchModel

from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.distributions import MultivariateNormal



class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, layer_depths = None):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', Linear(data_dim, 128))
        self.add_module('relu1', ReLU())
        self.add_module('linear2', Linear(128, 64))
        self.add_module('relu2', ReLU())
        self.add_module('linear3', Linear(64, 16))
        self.add_module('relu3', ReLU())
        self.add_module('linear5', Linear(16, 2))
 

class GPRegressionModel(GPyTorchModel, ExactGP):

    # Freeze everything but the last layer when training locally?
    def __init__(self, train_x, train_y, likelihood, architecture):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = ConstantMean()
        self.covar_module = GridInterpolationKernel(
            ScaleKernel(MaternKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )
        self.feature_extractor = architecture

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

    