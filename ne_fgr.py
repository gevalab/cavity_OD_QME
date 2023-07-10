import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import simpson

def da(tau_):
    return G * np.exp(1.j * w_da * tau_) * np.exp(-1.j * np.sum(.5*w_ * R_eq**2/hbar*np.sin(w_*tau_))) * np.exp(np.sum(.5*w_ * R_eq**2/hbar*1./np.tanh(.5*b*hbar*w_) * (np.cos(w_*tau_) - 1.) ))

def ad(tau_):
    return np.conj(da(tau_)) 

def daad_p(tau_):
    return G**2 * np.exp(1.j * w_da * (o1.t-tau_)) * np.exp(np.sum(.5*w_ * R_eq**2/hbar*1./np.tanh(.5*b*hbar*w_) * (np.cos(w_*(o1.t-tau_)) - 1.) )) * np.exp(-1.j* np.sum(.5*w_*R_eq**2/hbar*np.sin(w_*(o1.t - tau_))))

def conj_daad_p(tau_):
    return np.conj(daad_p(tau_)) 

def adda_p(tau_):
    return G**2 * np.exp(-1.j * w_da * (o1.t-tau_)) * np.exp(np.sum(.5*w_ * R_eq**2/hbar*1./np.tanh(.5*b*hbar*w_) * (np.cos(w_*(o1.t-tau_)) - 1.) )) * np.exp(1.j * np.sum(w_*R_eq**2/hbar*(np.sin(w_*o1.t) - np.sin(w_*tau_) - .5*np.sin(w_*(o1.t- tau_)) ) ))

def conj_adda_p(tau_):
    return np.conj(adda_p(tau_)) 

def dada_p(tau_):
    return G**2 * np.exp(1.j * w_da * (o1.t+tau_)) * np.exp(np.sum(.5*w_ * R_eq**2/hbar*1./np.tanh(.5*b*hbar*w_) * (2.*np.cos(w_*o1.t) + 2.*np.cos(w_*tau_) - 3. - np.cos(w_*(o1.t-tau_))  ) )) * np.exp(1.j* np.sum(w_*R_eq**2/hbar*(.5*np.sin(w_*(o1.t - tau_))-np.sin(w_*o1.t))))

def adad_p(tau_):
    return G**2 * np.exp(-1.j * w_da * (o1.t+tau_)) * np.exp(np.sum(.5*w_ * R_eq**2/hbar*1./np.tanh(.5*b*hbar*w_) * (2.*np.cos(w_*o1.t) + 2.*np.cos(w_*tau_) - 3. - np.cos(w_*(o1.t-tau_))  ) )) * np.exp(1.j* np.sum(w_*R_eq**2/hbar*(.5*np.sin(w_*(o1.t - tau_))+np.sin(w_*tau_))))

def dX(t, o):
    dxdt = np.zeros(o.shape, dtype = complex)
    X_o_ = np.array(X_o, dtype = complex).T
    tau = np.linspace(0,o1.t, X_o_[2][:].size )
    daad_ = np.array([daad_p(tau[k]) for k in range(tau.size)])
    adda_ = np.array([adda_p(tau[k]) for k in range(tau.size)])
    dada_ = np.array([dada_p(tau[k]) for k in range(tau.size)])
    adad_ = np.array([adad_p(tau[k]) for k in range(tau.size)])
    ad_ = np.array([ad(tau[k]) for k in range(tau.size)])
    da_ = np.array([da(tau[k]) for k in range(tau.size)])
    kdaad = 2./hbar**2 * simpson((daad_-da(o1.t)*ad_)*X_o_[1][:], tau).real
    kadda = 2./hbar**2 * simpson((adda_-ad(o1.t)*da_)*X_o_[0][:], tau).real
    kdaad_adda = 1./hbar**2 * simpson((daad_-da(o1.t)*ad_ + adda_-ad(o1.t)*da_)*X_o_[2][:], tau)
    kdada_adad = 1./hbar**2 * simpson((dada_-da(o1.t)*da_ + adad_-ad(o1.t)*ad_)*X_o_[3][:], tau)
    dxdt[1] = 2*np.real(1.j/hbar * da(o1.t)*o[2]) - kdaad + kadda
    dxdt[0] = -dxdt[1]
    dxdt[2] = 1.j/hbar * da(o1.t)*(o[1] - o[0]) - kdaad_adda + kdada_adad
    dxdt[3] = np.conj(dxdt[2])
    return dxdt

G = 30*0.00003674930882476
hbar = 1.
M = 4.529e8
w_c = 3.507e-4
eta = 1.066e6
omega = 3.507e-4
R_a = 2.929e-2
y0 = np.sqrt(M)*R_a/2
N = 101

w_da = 1168*0.00003674930882476
T = 300 # kelvin
kb = 3.167e-6 # Eh/K
b = 1./(kb*T)

N_s = N-1
w_max = 3*w_c
w_0 = w_c/N_s*(1.-np.exp(-w_max/w_c))
w = -w_c*np.log(1. - np.arange(1,N) * w_0/w_c)
c = np.sqrt(2.*eta*w_0/(M*np.pi))*w

D = np.zeros((N,N))
D[0,0] = omega**2 + np.sum(c**2/w**2)
D[0,1:] = c
D[1:,0] = c
di = np.diag_indices(N)[0][1:]
D[di,di] = w**2

w_2, v_ = np.linalg.eigh(D)

w_ = np.sqrt(w_2)

pm = np.zeros(N)
pm[0] = 1.
pm[1:] = -c/w**2

R_eq = 2.*y0 * v_.T @ pm

dt = 10.
t1 = 5e5

X_o = [] 
X = np.zeros(4, dtype = complex)
X[1] = 1.
X_o.append(X)
X_l = np.zeros((int(t1/dt), 3))

o1 = ode(dX).set_integrator('zvode')
o1.set_initial_value(X, 0)
o = 0
while o1.successful() and o1.t <t1:
    X_l[o,:] = [o1.t, (X[0]).real, (X[1]).real]
    X = o1.integrate(o1.t + dt)
    X_o.append(X)
    o += 1

np.savetxt('X.txt', X_l)
