# main.py
# Tests FFT and compares with other method

from ComputationalFinance import ComputationalFinance
import matplotlib.pyplot as plt
import numpy as np

# Parameters
S0 = 100.
K = 100.
T = 1.
r = .05
sigma = 0.2

# Object initiation
cf = ComputationalFinance(S0, K)

plt.figure(figsize=(20,14))
# Black-Scholes plot
import time
start = time.time()
callBS = []
for K in range(1, 200):
    answ = cf.Black_Scholes_European_Call(0, T, S0, K, r, 0., sigma)[0]
    callBS.append(answ)
end = time.time()
print("B-S: ", end - start)
plt.plot(callBS, 'x', label='BS')

# ... On top of it,
# FFT plot
import time
start = time.time()
callFFT = []
for K in range(1, 200):
    answ = cf.Option_Pricing_FFT_EO(S0, K, T, r, sigma)
    callFFT.append(answ)
end = time.time()
print("FFT: ", end - start)
plt.plot(callFFT, label="FFT")

# ... On top of it,
# Integrate plot
import time
start = time.time()
callInte = []
for K in range(1, 200):
    answ = cf.BS_call_integrate(S0, K, T, r, sigma)
    callInte.append(answ)
end = time.time()
print("Inte: ", end - start)
plt.plot(callInte, '+', label='Integral')
plt.xlabel("Strike Price")
plt.ylabel("Call value")
plt.legend(loc=0)
plt.show()
plt.savefig('BS, FFT, Integral.png')

# convert list to nparray
BS = np.asarray(callBS)
FFT = np.asarray(callFFT)
Inte = np.asarray(callInte)

# Error (minus)
plt.figure(figsize=(20,6))
errFFT = FFT - BS
errInte = Inte - BS
plt.plot(errFFT, label='Error of FFT')
plt.plot(errInte, '+', label='Error of Integral')
plt.xlabel("Strike Price")
plt.ylabel("Error (difference from Analytical value)")
plt.legend(loc=0)
plt.show()
plt.savefig('Minus_error.png')

# Absolute Relative error
plt.figure(figsize=(20,6))
areFFT = abs(FFT - BS) / abs(BS)
areInte = abs(Inte - BS) / abs(BS)
plt.plot(areFFT, label='ARE FFT')
plt.plot(areInte, '+', label='ARE Integral')
plt.xlabel("Strike Price")
plt.ylabel("ARE (from Analytical value)")
plt.legend(loc=0)
plt.show()
plt.savefig('ARE.png')


########## VG case comparison ##########

def CPUtime(S0, K, T, r, sigma, nu, theta):
    # Black-Scholes
    import time
    start = time.time()
    callBSv = []
    for K in range(1, 200):
        answ = cf.Black_Scholes_European_Call(0, T, S0, K, r, 0., sigma)[0]
        callBSv.append(answ)
    end = time.time()
    P = end - start  # analytic

    # FFT
    import time
    start = time.time()
    callFFTv = []
    for K in range(1, 200):
        answ = cf.VG_Option_Pricing_FFT_EO(S0, K, T, r, sigma, nu, theta)
        callFFTv.append(answ)
    end = time.time()
    FFT = end - start

    # # Computing delta and RNP
    import time
    start = time.time()
    callIntev = []
    for K in range(1, 200):
        answ = cf.VG_call_integrate(S0, K, T, r, sigma, nu, theta)
        callIntev.append(answ)
    end = time.time()
    PS = end - start   # Computing delta and RNP

    # convert list to nparray
    BSv = np.asarray(callBSv)
    FFTv = np.asarray(callFFTv)
    Intev = np.asarray(callIntev)

    # Error (minus)
   # P_FFT = FFTv - BSv
   # P_PS = Intev - BSv

    # Absolute Relative error
    P_FFT = abs(FFTv - BSv) / abs(BSv)
    P_PS = abs(Intev - BSv) / abs(BSv)
    #areFFTv.mean()
    #areIntev.mean()

    return FFT, PS, P, P_PS, P_FFT

###### TABLE: CPU TIMES FOR VG PRICING #######
# Storage
sigs = []
nus = []
thes = []
ts = []
FFTs = []
PSs = []
Ps = []

# Parameters
S0 = 100.
K = 100.
r = .05

# Case 1
sigma = 0.12
nu = .16
theta = -.33
T = 1.

FFT, PS, P, P_PS, P_FFT = CPUtime(S0, K, T, r, sigma, nu, theta)
sigs.append(sigma)
nus.append(nu)
thes.append(theta)
ts.append(T)
FFTs.append(FFT)
PSs.append(PS)
Ps.append(P)

# Case 2
sigma = .25
nu = 2.0
theta = -.10
T = 1.

FFT, PS, P, P_PS, P_FFT = CPUtime(S0, K, T, r, sigma, nu, theta)
sigs.append(sigma)
nus.append(nu)
thes.append(theta)
ts.append(T)
FFTs.append(FFT)
PSs.append(PS)
Ps.append(P)

# Case 3
sigma = .12
nu = .16
theta = -.33
T = .25

FFT, PS, P, P_PS, P_FFT = CPUtime(S0, K, T, r, sigma, nu, theta)
sigs.append(sigma)
nus.append(nu)
thes.append(theta)
ts.append(T)
FFTs.append(FFT)
PSs.append(PS)
Ps.append(P)

# Case 4
sigma = .25
nu = 2.0
theta = -.10
T = .25

FFT, PS, P, P_PS, P_FFT = CPUtime(S0, K, T, r, sigma, nu, theta)
sigs.append(sigma)
nus.append(nu)
thes.append(theta)
ts.append(T)
FFTs.append(FFT)
PSs.append(PS)
Ps.append(P)

# plot FIGURE
plt.figure(figsize=(10,3))
plt.plot(P_PS, label='ARE of VGPS')
plt.plot(P_FFT, label='ARE of FFT')
plt.title("Case 4 (p=100, r=.05, sig=.25, nu=2.0, theta=.10, t=.25)")
plt.legend(loc=0)
plt.xlabel("strike")
plt.ylabel("ARE")
plt.savefig("FIG2.png")
plt.show()

# plot TABLE
print("             Case 1             Case 2              Case 3              Case 4")
print(sigs)
print(nus)
print(thes)
print(ts)
print(FFTs)
print(PSs)
print(Ps)