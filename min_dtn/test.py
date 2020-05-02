import numpy as np
from scipy.special import hankel1,h1vp
import sys
sys.path.append('/home/janosch/projects/ngsandbox/opt/build')
# or append to PYTHONPATH ...
import min_dtn as m

Lmax = 72
omega = 16
R_min = 1.0
lam = np.array([mm**2 for mm in range(Lmax)]) 
lam_eval = np.array([mm for mm in range(100)],dtype=float)
val = np.zeros(len(lam_eval),dtype="complex")
dtn_ref = np.array([ -omega*h1vp(j,omega*R_min) / hankel1(j,omega*R_min) for j in range(Lmax) ]) 
scale_w = 10**6
alpha_decay = 1.5
weights = np.array([scale_w*np.exp(-mm/alpha_decay) for mm in range(Lmax)])

np.random.seed(123)

l1_guess = np.random.rand(1,1) + 1j*np.random.rand(1,1)
l2_guess = np.random.rand(1,1) + 1j*np.random.rand(1,1)
l_dtn = m.learned_dtn(lam,dtn_ref,weights**2)
#ansatz = "full"
ansatz = "minimalIC"
flags = {"max_num_iterations":50000,
         "check_gradients":False, 
         "use_nonmonotonic_steps":True,
         "minimizer_progress_to_stdout":True,
         "num_threads":4,
         "report_level":"Brief"}
final_res = np.zeros(2*len(lam))
for N in range(1,8): 
    l_dtn.Run(l1_guess,l2_guess,ansatz,flags)
    print("l1 = ", l1_guess)
    print("l2 = ", l2_guess)
    m.eval_dtn_fct(l1_guess,l2_guess,lam_eval,val)
    print("val = ", val)
    #print("final_res = ", final_res)
    l1_old = l1_guess.copy()
    l2_old = l2_guess.copy()
    l1_guess = np.zeros((N+1,N+1),dtype='complex')
    l2_guess = np.zeros((N+1,N+1),dtype='complex')
    
    if ansatz in ["medium","full"]:
        l1_guess = 1e-3*(np.random.rand(N+1,N+1) + 1j*np.random.rand(N+1,N+1))
        l2_guess = 1e-3*(np.random.rand(N+1,N+1) + 1j*np.random.rand(N+1,N+1))
        l1_guess[:N,:N] = l1_old[:]
        l2_guess[:N,:N] = l2_old[:]
        l1_guess[N,N] = 1.0
    elif ansatz == "minimalIC":
        l1_guess = 1e-3*(np.random.rand(N+1,N+1) + 1j*np.random.rand(N+1,N+1))
        l1_guess[:N,:N] = l1_old[:]
        l2_guess[:N,:N] = l2_old[:]
        l1_guess[N,N] = -100-100j
        l2_guess[0,N] = 1e-3*(np.random.rand(1) + 1j*np.random.rand(1))
    else:
        l1_guess[:N,:N] = l1_old[:]
        l2_guess[:N,:N] = l2_old[:]
        l1_guess[N,N] = -100-100j
        l1_guess[0,N] = l1_guess[N,0] = 10**(3/2)*(np.random.rand(1) + 1j*np.random.rand(1))
        #l1_guess[0,N] = l1_guess[N,0] = 1.0

    input("continue")

