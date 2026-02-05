import os, sys

sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from TT_helpers import generate_tt_cores, matricize_tt_cores, compute_G_left, compute_G_right
from functions.utils import pure_power_features_full


R = 3 
M = 10
D= 7
N = 100

X = np.random.rand(N, D)
W_D = generate_tt_cores(D=D, R=R, M=M)
Phi = pure_power_features_full(X, M)

G_left = compute_G_left(Phi[:3], W_D[:3])
G_right = compute_G_right(Phi[3:], W_D[3:]) 
print(G_left)