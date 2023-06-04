import sys

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

param_grid = {'tol': tol, 
                'max_iter': max_iter}