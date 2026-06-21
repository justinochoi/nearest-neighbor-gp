import arviz as az 
import pymc as pm 
import pytensor 
import pytensor.tensor as pt 
import numpy as np 
import xarray as xr 
from scipy.spatial import KDTree 
from scipy.spatial.distance import cdist 

def get_NNind(coords, m): 
    n = len(coords) 
    NNind = np.full((n-1, m), -1) 

    for i in range(1, n): 
        predecessors = coords[:i] 
        tree = KDTree(predecessors)

        k = min(m, i) 
        _, idx = tree.query(coords[i], k=k) 
        NNind[i-1, :k] = idx 

    return NNind

def get_NNdist(coords, NNind, m): 
    n = len(coords) 
    distance = np.full((n-1, m), 0.0)

    for i in range(1, n): 
        k = min(m, i)
        neighbors = coords[NNind[i-1, :k]] 
        # correct axis? 
        dists = np.linalg.norm(neighbors - coords[i], axis=1) 
        distance[i-1, :k] = dists 

    return distance 

def get_NNdistM(coords, NNind, m): 
    n = len(coords) 
    lower_t = np.full((n-1, int(m*(m-1)/2)), 0.0)

    for i in range(1, n): 
         k = min(m, i) 
         neighbors = coords[NNind[i-1, :k]] 
         D = cdist(neighbors, neighbors)
         rows, cols = np.tril_indices(k, k=-1)
         vals = D[rows, cols] 
         lower_t[i-1, :len(vals)] = vals 

    return lower_t 

def make_synthetic_data(n, beta=[1,2], tau=1.0, sigma=1.0, ell=0.3, seed=76): 
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size = (n, 2)) 
    D = cdist(coords, coords) 
    C = sigma**2 * np.exp(-D / ell) # exponential quadratic 
    C += 1e-6 * np.eye(n) # numerical stability 

    intercept = np.ones((n,1)) 
    predictor = rng.normal(0, 1, size=(n,1)) 

    X = np.concatenate((intercept, predictor), axis=1)
    w = rng.multivariate_normal(np.zeros(n), C) 
    y = rng.normal(np.dot(X, beta) + w, tau)

    return coords, C, X, y

def nngp_response_step(NNdist_i, NNdistM_i, NNind_i, mask_i, m, 
                        resid, sigma, tau, ell, tril_rows, tril_cols):
    
    kappa = tau**2 / sigma**2 + 1
    
    off_diag = pt.exp(-NNdistM_i / ell)
    
    iNNdistM = pt.zeros((m, m))  # m is a plain Python int here
    iNNdistM = pt.set_subtensor(iNNdistM[tril_rows, tril_cols], off_diag)
    iNNdistM = iNNdistM + iNNdistM.T
    
    mask_2d = mask_i[:, None] * mask_i[None, :]
    iNNdistM = iNNdistM * mask_2d
    iNNdistM = pt.set_subtensor(iNNdistM[pt.arange(m), pt.arange(m)],
                                 pt.where(mask_i > 0, kappa, 1.0))
    
    iNNCholL = pt.linalg.cholesky(iNNdistM)
    iNNcorr = pt.exp(-NNdist_i / ell) * mask_i
    
    v = pt.linalg.solve_triangular(iNNCholL, iNNcorr, lower=True)
    V_i = kappa - pt.dot(v, v)
    
    v2 = pt.linalg.solve_triangular(iNNCholL, v, lower=True, trans='T')
    U_i_update = pt.dot(v2, resid[NNind_i] * mask_i)

    return V_i, U_i_update

def nngp_response_logp_pt(X, y, m, beta, sigma, tau, ell,
                           NNind, NNdist, NNdistM):
    n = y.shape[0]
    resid = y - pt.dot(X, beta)
    kappa = tau**2 / sigma**2 + 1

    NNind_const = pt.constant(NNind.astype(int))
    NNdist_const = pt.constant(NNdist)
    NNdistM_const = pt.constant(NNdistM)
    mask = (NNind != -1).astype(float)
    mask_const = pt.constant(mask)
    
    tril_rows_const = pt.constant(np.tril_indices(m, k=-1)[0])
    tril_cols_const = pt.constant(np.tril_indices(m, k=-1)[1])
    
    (V_vals, U_updates), _ = pytensor.scan(
        fn=nngp_response_step,
        sequences=[NNdist_const, NNdistM_const, NNind_const, mask_const],
        non_sequences=[m, resid, sigma, tau, ell, tril_rows_const, tril_cols_const],
    )
    
    U = pt.concatenate([[resid[0]], resid[1:] - U_updates])
    V = pt.concatenate([[kappa], V_vals])
    
    return -0.5 * (1/sigma**2 * pt.dot(U, U/V) + pt.sum(pt.log(V)) + n*pt.log(sigma**2))

# only runs when file is executed directly 
if __name__ == '__main__': 
    # toy example 
    n = 5000
    m = 10 
    sigma = 1.0
    tau = 1.0
    ell = 0.3
    beta = [1,2]

    coords, C, X, y = make_synthetic_data(
        n, beta, tau, sigma, ell, seed = 76
    ) 
    order = np.argsort(coords[:, 0]) 
    coords_sorted = coords[order] 
    y_sorted = y[order]
    X_sorted = X[order] 

    NNind = get_NNind(coords_sorted, m=10) 
    NNdist = get_NNdist(coords_sorted, NNind, m=10) 
    NNdistM = get_NNdistM(coords_sorted, NNind, m=10)

    with pm.Model() as nngp: 
    
        beta = pm.Normal('beta', mu=0, sigma=1, shape=2)
        sigma = pm.InverseGamma('sigma', alpha=3, beta=1)
        ell = pm.Gamma('ell', alpha=2, beta=2)
        tau = pm.HalfNormal('tau', sigma=1) 

        pm.Potential(
            'nngp', 
            nngp_response_logp_pt(
                X_sorted, y_sorted, m, beta, sigma, tau, ell, 
                NNind, NNdist, NNdistM
            )
        )

    with nngp: 
        nngp_trace = pm.sample(chains=4, cores=4, random_seed=76, nuts_sampler='nutpie')

    nngp_trace.to_netcdf('toy_example_nngp_trace.nc')
    az.summary(nngp_trace)