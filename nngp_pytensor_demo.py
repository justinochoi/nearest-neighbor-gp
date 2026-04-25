import arviz as az 
import pymc as pm 
import pytensor 
import pytensor.tensor as pt 
import numpy as np 
from scipy.spatial import KDTree 
from scipy.spatial.distance import cdist 

### helper functions ### 
def build_neighbor_array(coords, m): 
    n = len(coords) 
    neighbor_idx = np.full((n, m), -1) 

    for i in range(1, n): 
        predecessors = coords[:i] 
        tree = KDTree(predecessors)

        k = min(m, i) 
        _, idx = tree.query(coords[i], k=k) 
        neighbor_idx[i, :k] = idx 

    return neighbor_idx 

def exp_cov(coords_a, coords_b, sigma2, ell): 
    # assume a is (n x 2) and b is (m x 2)
    # so diff is (n x m x 2)
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    # clipping to prevent negative values 
    D = pt.sqrt(pt.clip(pt.sum(pt.sqr(diff), axis=2), 0, np.inf)) 

    return sigma2 * pt.exp(-D / ell)

def make_synthetic_data(n, sigma2=1.0, ell=0.3, seed=76): 
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size = (n, 2)) 
    D = cdist(coords, coords) 
    C = sigma2 * np.exp(-D / ell) # exponential quadratic 
    C += 1e-6 * np.eye(n) # numerical stability 
    w = rng.multivariate_normal(np.zeros(n), C) 

    return coords, w, C

### generating data ### 
coords, w, C = make_synthetic_data(n=50) 
order = np.argsort(coords[:, 0]) 
coords_sorted = coords[order] 
w_sorted = w[order] 
neighbor_idx = build_neighbor_array(coords=coords_sorted, m=10)
neighbor_idx_safe = np.where(neighbor_idx == -1, 0, neighbor_idx).astype(int)
neighbor_mask = (neighbor_idx != -1).astype(float) 

def B_and_F_step(coords_i, neighbor_idx_i, mask_i, coords, sigma2, ell):
    m = neighbor_idx_i.shape[0] 
    
    # covariance computations - always use all m neighbors
    C_ii = exp_cov(coords_i[None], coords_i[None], sigma2, ell)
    C_i_neighbor = exp_cov(coords_i[None], coords[neighbor_idx_i], sigma2, ell)  # (1, m)
    C_neighbor_i = exp_cov(coords[neighbor_idx_i], coords_i[None], sigma2, ell)  # (m, 1)
    C_neighbor_neighbor = exp_cov(coords[neighbor_idx_i], coords[neighbor_idx_i], sigma2, ell)  # (m, m)
    
    # mask out invalid neighbor contributions
    mask_2d = mask_i[:, None] * mask_i[None, :]  # (m, m)
    C_i_neighbor_masked = C_i_neighbor * mask_i[None, :]
    C_neighbor_i_masked = C_neighbor_i * mask_i[:, None]
    C_neighbor_neighbor_masked = C_neighbor_neighbor * mask_2d
    
    # keep matrix invertible for invalid entries
    C_neighbor_neighbor_safe = (
        C_neighbor_neighbor_masked 
        + pt.diag(1 - mask_i)
        + 1e-6 * pt.eye(m)
    )
    
    C_nn_inv = pt.linalg.inv(C_neighbor_neighbor_safe)
    
    B_i = (C_i_neighbor_masked @ C_nn_inv).ravel() * mask_i
    F_i = C_ii - (C_i_neighbor_masked @ C_nn_inv @ C_neighbor_i_masked)
    
    return B_i, F_i

def compute_B_and_F(coords_sorted, neighbor_idx_safe, neighbor_mask, sigma2, ell):
    coords_const = pt.constant(coords_sorted)
    neighbor_idx_const = pt.constant(neighbor_idx_safe)
    mask_const = pt.constant(neighbor_mask) 
    (B, F), _ = pytensor.scan(
        fn=B_and_F_step,
        sequences=[coords_const, neighbor_idx_const, mask_const],
        non_sequences=[coords_const, sigma2, ell]
    )
    return B, F

def w_step(i, w_raw_i, neighbor_idx_i, mask_i, B_i, F_i, w_prev):
    w_i = pt.dot(B_i * mask_i, w_prev[neighbor_idx_i]) + pt.sqrt(F_i) * w_raw_i
    w_new = pt.set_subtensor(w_prev[i], w_i)
    return w_new

def transform_w(w_raw, neighbor_idx_safe, neighbor_mask, B, F, n):
    indices_const = pt.constant(np.arange(n, dtype=int))
    neighbor_idx_const = pt.constant(neighbor_idx_safe)
    mask_const = pt.constant(neighbor_mask) 
    w_init = pt.zeros(n)
    
    w, _ = pytensor.scan(
        fn=w_step,
        sequences=[indices_const, w_raw, neighbor_idx_const, mask_const, B, F],
        outputs_info=[w_init]
    )
    return w[-1]

### building the model ### 
with pm.Model() as nngp_model: 

    n = len(coords_sorted)
    sigma2 = pm.InverseGamma('sigma2', alpha=3, beta=1)
    ell = pm.Gamma('ell', alpha=2, beta=2)
    w_raw = pm.Normal('w_raw', mu=0, sigma=1, shape=n)
    B, F = compute_B_and_F(coords_sorted, neighbor_idx_safe, 
                                 neighbor_mask, sigma2, ell)
    w = pm.Deterministic('w', transform_w(w_raw, neighbor_idx_safe, 
                                          neighbor_mask, B, F, n))

pm.model_to_graphviz(nngp_model)

with nngp_model:
    trace = pm.sample(
        draws=500, tune=500, chains=4, cores=4, 
        random_seed=76, nuts_sampler='numpyro', 
        target_accept=0.95
    )

az.summary(trace, var_names=['sigma2','ell'])
az.plot_pair(trace, var_names=['sigma2','ell'])

### compare with GP ### 
with pm.Model() as full_gp: 

    sigma2 = pm.InverseGamma('sigma2', alpha=3, beta=1)
    ell = pm.Gamma('ell', alpha=2, beta=2)
    cov_func = sigma2 * pm.gp.cov.Matern12(2, ls=ell)
    gp = pm.gp.Latent(cov_func=cov_func)
    w = gp.prior('w', X=coords_sorted)

pm.model_to_graphviz(full_gp)

with full_gp: 
    gp_trace = pm.sample(
        draws=500, tune=500, chains=4, cores=4, 
        random_seed=76, nuts_sampler='numpyro', 
        target_accept=0.95
    )

az.summary(gp_trace, var_names=['sigma2','ell'])
