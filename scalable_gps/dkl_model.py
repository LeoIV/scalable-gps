import math
import tqdm
import torch
import gpytorch

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, layer_depths = None):
        self.add_module('linear1', torch.nn.Linear(data_dim, 256))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(256, 128))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(128, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(64, 16))        
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('linear5', torch.nn.Linear(16, 2))
 

class GPRegressionModel(gpytorch.models.ExactGP):

    # Freeze everything but the last layer when training locally?
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.Matern(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )
        self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
