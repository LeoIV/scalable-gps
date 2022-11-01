import torch
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine


def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
                torch.rand(n_candidates, dim)
                <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), )] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


def optimize_llhood(model, train_x, train_y, mll, num_steps=50):
    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    cum_loss = 0
    for _ in range(
            num_steps
    ):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        cum_loss += loss
        loss.backward()
        optimizer.step()
