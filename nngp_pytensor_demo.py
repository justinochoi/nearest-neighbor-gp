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

def make_synthetic_data(n=50, sigma2=1.0, ell=0.3, seed=76): 
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size = (n, 2)) 
    D = cdist(coords, coords) 
    C = sigma2 * np.exp(-D / ell) # exponential quadratic 
    C += 1e-6 * np.eye(n) # numerical stability 
    w = rng.multivariate_normal(np.zeros(n), C) 

    return coords, w, C

### generating data ### 
coords, w, C = make_synthetic_data() 
order = np.argsort(coords[:, 0]) 
coords_sorted = coords[order] 
w_sorted = w[order] 
neighbor_idx = build_neighbor_array(coords=coords_sorted, m=10)

### functions required to evaulate NNGP ### 
def compute_B_and_F(coords, neighbor_idx, sigma2, ell): 
    n, m = neighbor_idx.shape 
    B_rows = [] 
    F_vals = [] 

    for i in range(n): 
        idx = neighbor_idx[i][neighbor_idx[i] != -1] 
        k = len(idx) 

        if k == 0: 
            B_rows.append(pt.zeros(m))
            F_vals.append(exp_cov(coords[[i]], coords[[i]], sigma2, ell)[0,0]) # scalar 
            continue 

        C_ii = exp_cov(coords[[i]], coords[[i]], sigma2, ell)[0,0] 
        C_i_neighbor = exp_cov(coords[[i]], coords[idx], sigma2, ell) 
        C_neighbor_neighbor = exp_cov(coords[idx], coords[idx], sigma2, ell)

        B_i_vals = pt.linalg.solve(C_neighbor_neighbor.T, C_i_neighbor.T).T.ravel() 
        B_i_padded = pt.concatenate([B_i_vals, pt.zeros(m - k)])
        B_rows.append(B_i_padded) 

        F_i = C_ii - (C_i_neighbor @ pt.linalg.solve(C_neighbor_neighbor, C_i_neighbor.T))[0, 0] 
        F_vals.append(F_i) 

    B = pt.stack(B_rows) 
    F = pt.stack(F_vals) 

    return B, F 

def nngp_logp(w, neighbor_idx, B, F): 
    log_lik = 0 

    for i in range(neighbor_idx.shape[0]): # TensorVariables have no len() 
        idx = neighbor_idx[i][neighbor_idx[i] != -1] 
        resid = w[i] - B[i, :len(idx)] @ w[idx]
        term = -0.5 * pt.log(2*pt.pi) - 0.5 * pt.log(F[i]) - 0.5 * pt.square(resid) / F[i] 
        log_lik += term 

    return log_lik 


### building the model ### 
with pm.Model() as nngp_model: 

    sigma2 = pm.HalfNormal('sigma2', sigma=2) 
    ell = pm.Gamma('ell', alpha=2, beta=2)
    w = pm.Normal('w', mu=0, sigma=1, shape=len(coords_sorted))

    B, F = compute_B_and_F(coords_sorted, neighbor_idx, sigma2, ell) 

    pm.Potential('nngp', nngp_logp(w, neighbor_idx, B, F)) 

pm.model_to_graphviz(nngp_model)

with nngp_model:
    trace = pm.sample(100, tune=100, chains=1, random_seed=76, nuts_sampler='numpyro')
# 0.04 step size with 92% acceptance prob! 




### tests ### 
# test that the exp_cov function works properly 
sigma2_test = pt.scalar('sigma2')
ell_test = pt.scalar('ell')

C_test = exp_cov(coords_sorted[:3], coords_sorted[:3], sigma2_test, ell_test)
f = pytensor.function([sigma2_test, ell_test], C_test)
print(f(1.0, 0.3))



        
