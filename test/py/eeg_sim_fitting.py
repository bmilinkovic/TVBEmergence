import sys
import time
from autograd import numpy as np, grad
from autograd.misc.optimizers import adam, rmsprop, sgd
from scipy.optimize import minimize
import pylab as pl


# simplification of the generic 2d oscillator
def dfun(state, w, theta):
    x, y = state
    tau, a, c, f = theta
    dx = tau * (x - x**3.0 / 3.0 + y)
    dy = (1.0 / tau) * (a - x + c * np.dot(w, x))
    return np.array([dx, dy]) / (f + 10)

# eeg spectrum forward model
def fwd(gain, state, win):
    eeg_win = np.dot(gain, state[0]).reshape((gain.shape[0], -1, win.size))
    eeg_spec = np.abs(np.fft.fft(eeg_win * win, axis=-1)).mean(axis=1) # (n_eeg, win.size)
    return eeg_spec

# prediction error of model
neval = [0] # yuck but whatev
def make_loss(dt, w, eeg, gain, win, dw):
    n_node = len(w)
    n_eeg, _ = eeg.shape
    idw = 1.0 / dw
    def pred(params):
        # state.shape = (2, n_node, n_time)
        theta = params[-4:]
        state = np.reshape(params[:-4], (2, n_node, -1))
        # predict latent state evolution
        next_state = state + dt * dfun(state, w, theta)
        # predict observed data
        eeg_hat = fwd(gain, state, win)
        return next_state, eeg_hat
    def loss(params):
        neval[0] += 1
        if neval[0] % 10 == 0:
            print('.', end=''); sys.stdout.flush()
        next_state, eeg_hat = pred(params)
        loss_state = np.sum(((next_state[:, :, :-1] - state[:, :, 1:]) * idw)**2)
        loss_eeg = np.sum((eeg - eeg_hat)**2)
        return loss_eeg + loss_state
    return loss, pred

# create test data
#n_node, n_time, n_eeg = 84, 4800, 64
n_node, n_time, n_eeg = 84, 10000, 64
dt, sig = 0.1, 0.03
dw = np.sqrt(dt) * sig
theta = tau, a, c, f = 3.0, 0.7, 0.1, 1.0
state = np.random.randn(2, n_node, n_time)
gain = np.random.rand(n_eeg, n_node) / n_node
w = np.random.rand(n_node, n_node) / n_node
eeg = np.zeros((n_eeg, n_time))
eeg[:, 0] = gain.dot(state[0, :, 0])
for t in range(n_time - 1):
    dW_t = dw * state[..., t + 1]
    state[..., t + 1] = state[..., t] + dt * dfun(state[..., t], w, theta) + dW_t
    eeg[:, 0] = gain.dot(state[0, :, t + 1])

# spectral analysis of eeg data
n_win = 1
win = np.blackman(eeg.shape[-1]//n_win)
eeg = fwd(gain, state, win)

# make loss & grad, note starting loss
loss, pred = make_loss(dt, w, eeg, gain, win, dw)
gl = grad(loss)
print('ll truth %0.3f' % (np.log(loss(np.r_[state.reshape((-1, )), np.array(theta)]))))

# perturb known states for initial guess on optimizers
state_ = state + np.random.randn(*state.shape)
theta_ = np.array(theta) + np.random.randn(4)
x0_ = np.r_[state_.reshape((-1, )), theta_]

# run different optimizers for certain number of iterations
# and compare performance (in reduction of log loss (rrl) per loss eval)
max_iter = 1000
x0s = {}
for opt in 'adam rmsprop bfgs tnc'.split():
    tic = time.time()
    print(opt.rjust(8), end=': ')
    x0 = x0_.copy()
    neval[0] = 0
    ll0 = np.log(loss(x0))
    print('ll %0.3f' % ll0, end=' ')
    if opt in ('bfgs', 'tnc'):
        method = {'bfgs': 'L-BFGS-B', 'tnc': 'TNC'}[opt]
        for i in range(3):
            x0 = minimize(loss, x0, method=method, jac=gl, options={'maxiter': max_iter//3}).x
    elif opt in ('adam', 'rmsprop'):
        cb = lambda x, i: gl(x)
        for h in [0.1, 0.01, 0.001]:
            x0 = eval(opt)(cb, x0, step_size=h, num_iters=max_iter//3)
    else:
        raise ValueError(opt)
    x0s[opt] = x0.copy()
    toc = time.time() - tic
    ll1 = np.log(loss(x0))
    rll_eval = neval[0] / (ll0 - ll1)
    print(' %0.3f, %d feval, %0.3fs, %0.3f evals/rll' % (ll1, neval[0], toc, rll_eval))

# check optimized spectra against that for known parameters
pl.figure(figsize=(10, 10))
pl.subplot(211)
pl.plot(np.r_[:n_time]*dt*1e-3, state[0].T + np.r_[:n_node], 'k', alpha=0.2)
pl.subplot(212)
Fs = np.fft.fftfreq(win.size, dt*1e-3)
Fsm = (Fs >= 0) * (Fs < 150)
pl.loglog(Fs[Fsm], eeg.mean(axis=0)[Fsm], 'k', alpha=0.7)
names = ['sim']
_, eeg_h = pred(x0_)
pl.loglog(Fs[Fsm], eeg_h.mean(axis=0)[Fsm], alpha=0.2)
names.append('sim perturb')
for opt, x0 in x0s.items():
    _, eeg_h = pred(x0)
    pl.loglog(Fs[Fsm], eeg_h.mean(axis=0)[Fsm], alpha=0.2)
    names.append(opt)
pl.legend(names)
pl.savefig('dfuns-spectra-opt-change.png', dpi=200)
import os; os.system('open dfuns-spectra-opt-change.png')