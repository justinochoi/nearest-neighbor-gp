import pytensor
import pytensor.tensor as pt
import numpy as np

# test inputs as plain numpy
coords_test = coords_sorted[:5]
neighbor_idx_test = neighbor_idx_safe[:5]
mask_test = neighbor_mask[:5]

sigma2_test = pt.dscalar('sigma2')
ell_test = pt.dscalar('ell')

n = 5  # plain python int

coords_const = pytensor.tensor.constant(coords_test)
neighbor_idx_const = pytensor.tensor.constant(neighbor_idx_test)
mask_const = pytensor.tensor.constant(mask_test)

(B_test, F_test), _ = pytensor.scan(
    fn=B_and_F_step,
    sequences=[coords_const, neighbor_idx_const, mask_const],
    non_sequences=[coords_const, sigma2_test, ell_test]
)

f = pytensor.function([sigma2_test, ell_test], [B_test, F_test])
print(f(1.0, 0.3))

# test that the exp_cov function works properly 
sigma2_test = pt.scalar('sigma2')
ell_test = pt.scalar('ell')

C_test = exp_cov(coords_sorted[:3], coords_sorted[:3], sigma2_test, ell_test)
f = pytensor.function([sigma2_test, ell_test], C_test)
print(f(1.0, 0.3))


sigma2_sym = pt.dscalar('sigma2')
ell_sym = pt.dscalar('ell')

B_loop, F_loop = compute_B_and_F(coords_sorted, neighbor_idx, sigma2_sym, ell_sym)
B_scan, F_scan = compute_B_and_F_scan(coords_const, neighbor_idx_const, mask_const, sigma2_sym, ell_sym)

f_loop = pytensor.function([sigma2_sym, ell_sym], [B_loop, F_loop])
f_scan = pytensor.function([sigma2_sym, ell_sym], [B_scan, F_scan])

B_loop_val, F_loop_val = f_loop(1.0, 0.3)
B_scan_val, F_scan_val = f_scan(1.0, 0.3)

print(np.allclose(B_loop_val, B_scan_val, atol=1e-4))
print(np.allclose(F_loop_val, F_scan_val, atol=1e-5))

print("B max difference:", np.max(np.abs(B_loop_val - B_scan_val)))
print("F max difference:", np.max(np.abs(F_loop_val - F_scan_val)))

# look at first few rows
print("\nB loop first 5 rows:\n", B_loop_val[:5])
print("\nB scan first 5 rows:\n", B_scan_val[:5])

print("\nF loop:", F_loop_val[:5])
print("\nF scan:", F_scan_val[:5])

print("neighbor_idx first 5 rows:\n", neighbor_idx[:5])
print("neighbor_idx_safe first 5 rows:\n", neighbor_idx_safe[:5])

# check which rows differ
for i in range(len(F_loop_val)):
    if not np.allclose(F_loop_val[i], F_scan_val[i], atol=1e-5):
        print(f"Row {i} differs:")
        print(f"  loop: {F_loop_val[i]}")
        print(f"  scan: {F_scan_val[i]}")
        print(f"  neighbor_idx: {neighbor_idx[i]}")
        print(f"  neighbor_idx_safe: {neighbor_idx_safe[i]}")
