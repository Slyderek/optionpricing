#ComputationalFinance.py
from math import sqrt, exp, log
from scipy.stats import norm
import numpy as np

class ComputationalFinance ( object ):
    def __init__( self, stock_price, strike_price ):
        self.stock_price  = stock_price
        self.strike_price = strike_price

    def Gaussian_RBF(self, eps, x, y):
        r = (x - y) ** 2
        phi_ga_rbf = np.exp(-eps ** 2 * r ** 2)
        phi_x_ga_rbf = -2*eps**2 * r * np.exp(-eps**2 * r**2)
        phi_xx_ga_rbf = (4*eps**4 * r**2 - 2*eps**2) * np.exp(-eps**2 * r**2)
        return phi_ga_rbf, phi_x_ga_rbf, phi_xx_ga_rbf

    def Multiquadratic_RBF(self, eps, x, y):
        r = (x - y) ** 2
        phi_mq_rbf = np.sqrt(1 + (eps * r) ** 2)
        phi_x_mq_rbf = eps**2 * r / phi_mq_rbf
        phi_xx_mq_rbf = eps**2 / phi_mq_rbf - eps**4 * r**2 / phi_mq_rbf**3
        return phi_mq_rbf, phi_x_mq_rbf, phi_xx_mq_rbf

    def Inverse_Multiquadratic_RBF(self, eps, x, y):
        r = (x - y) ** 2
        phi_imq_rbf = 1 / (np.sqrt(1 + (eps * r) ** 2))
        phi_x_imq_rbf = -eps**2 * r/ ((1 + (eps * r) ** 2) ** (3/2))
        phi_xx_imq_rbf = 3*eps**4 * r**2 / ((1 + (eps * r) ** 2) ** (5/2)) \
                            - eps**2 / ((1 + (eps * r) ** 2) ** (3/2))
        return phi_imq_rbf, phi_x_imq_rbf, phi_xx_imq_rbf

    def Inverse_Quadratic_RBF(self, eps, x, y):
        r = (x - y) ** 2
        phi_iq_rbf = 1 / (1 + (eps * r) ** 2)
        phi_x_iq_rbf = -2*eps**2 * r / (1 + (eps * r) ** 2)
        phi_xx_iq_rbf = 2*eps**2 * (3*eps**2 * r**2 - 1) / (eps**2 * r**2 + 1)**3
        return phi_iq_rbf, phi_x_iq_rbf, phi_xx_iq_rbf

    def European_Call_Option_Payoff( self ):
        european_call_payoff = max(0, self.stock_price - self.strike_price)
        return european_call_payoff
        # object (mesh possible) that has z value

    def European_Put_Option_Payoff( self ):
        european_put_payoff = max(0, self.strike_price - self.stock_price)
        return european_put_payoff

    '''
    use control flow to judge whether their necessary conditions hold. 
    If they do not hold, please print out a warning and quit.
    '''
    def Bull_Call_Spread(self, S, K1, K2):
        if not (K1 < K2):
            print("Warning: Bull call spread inputs are incorrect")
            print("strike_price_1 < strike_price_2 does not hold")
            quit()
        else:
            # compute
            call1 = ComputationalFinance(S,K1)
            call2 = ComputationalFinance(S,K2)
            bull_call_spread_payoff = call1.European_Call_Option_Payoff()\
                                      - call2.European_Call_Option_Payoff()
            return bull_call_spread_payoff

    def Bull_Put_Spread(self, S, K1, K2):
        if not (K1 > K2):
            print("Warning: Bull put spread inputs are incorrect")
            print("strike_price_1 > strike_price_2 does not hold")
            quit()
        else:
            put1 = ComputationalFinance(S,K1)
            put2 = ComputationalFinance(S,K2)
            bull_put_spread_payoff = put1.European_Put_Option_Payoff()\
                                      - put2.European_Put_Option_Payoff()
        return bull_put_spread_payoff

    def Bear_Call_Spread(self, S, K1, K2):
        if not (K1 > K2):
            print("Warning: Bear call spread inputs are incorrect")
            print("strike_price_1 > strike_price_2 does not hold")
            quit()
        else:
            call1 = ComputationalFinance(S, K1)
            call2 = ComputationalFinance(S, K2)
            bear_call_spread_payoff = call1.European_Call_Option_Payoff()\
                                      - call2.European_Call_Option_Payoff()
        return bear_call_spread_payoff

    def Collar(self, S, K1, K2):
        if not (K1 < K2):
            print("Warning: Collar inputs are incorrect")
            print("strike_price_1 < strike_price_2 does not hold")
            quit()
        else:
            put = ComputationalFinance(S, K1)
            call = ComputationalFinance(S, K2)
            collar_payoff = put.European_Put_Option_Payoff()\
                                      - call.European_Call_Option_Payoff()
        return collar_payoff

    def Straddle(self, S, K):
        call = ComputationalFinance(S, K)
        put = ComputationalFinance(S, K)
        straddle_payoff = call.European_Call_Option_Payoff()\
                                  + put.European_Put_Option_Payoff()
        return straddle_payoff

    def Strangle(self, S, K1, K2):
        if not (K1 != K2):
            print("Warning: Strangle inputs are incorrect")
            print("strike_price_1 != strike_price_2 does not hold")
            quit()
        else:
            call = ComputationalFinance(S, K1)
            put = ComputationalFinance(S, K2)
            strangle_payoff = call.European_Call_Option_Payoff() +\
                            put.European_Put_Option_Payoff()
        return strangle_payoff

    def Butterfly_Spread(self, S, p2, K1, K2, K3):
        '''
        :param S: Stock price
        :param p2: the units of selling for call option with strike price K2
        :param K1<K2<K3 should be met
        '''
        if not (K1 < K2 < K3):
            print("Warning: Butterfly spread inputs are incorrect")
            print("strike_price_1 < strike_price_2 < strike_price_3 does not hold")
            quit()
        else:
            # compute the butterfly spread
            l = (K3-K2)/(K3-K1)  # lambda
            call1 = ComputationalFinance(S, K1)
            call2 = ComputationalFinance(S, K2)
            call3 = ComputationalFinance(S, K3)
            butterfly_spread_payoff = p2 *(l*call1.European_Call_Option_Payoff()+\
                                        (1-l)*call3.European_Call_Option_Payoff()-\
                                              call2.European_Call_Option_Payoff() )
            return butterfly_spread_payoff

    def d1(self, S, K, r, div, vol, T, t):
        return (np.log(S/K) + (r-div + (vol**2)/2 ) * (T-t)) / (vol * np.sqrt(T-t))

    def d2(self, S, K, r, div, vol, T, t):
        return self.d1(S, K, r, div, vol, T, t) - vol * np.sqrt(T-t)

    def N(self, d):
        return norm.cdf(d)

    def Black_Scholes_European_Call(self, t, maturity_date, stock_price,\
        strike_price, interest_rate, dividend_yield, volatility):
        T = maturity_date; S = stock_price; K = strike_price
        r = interest_rate; div = dividend_yield; vol = volatility
        bs_european_call_price = np.exp(-div*(T-t)) * S * self.N(self.d1(S,K,r,div,vol,T,t))\
                                 - np.exp(-r*(T-t)) * K * self.N(self.d2(S,K,r,div,vol,T,t))
        bs_european_call_delta = np.exp(-div*(T-t)) * self.N(self.d1(S,K,r,div,vol,T,t))
        bs_european_call_theta = div*np.exp(-div*(T-t))*S*self.N(self.d1(S,K,r,div,vol,T,t))\
                                 - (vol*np.exp(-div*(T-t))*S*norm.pdf(self.d1(S,K,r,div,vol,T,t)))\
                                 / (2 * np.sqrt(T-t))\
                                 - r*np.exp(-r*(T-t))*K*self.N(self.d2(S,K,r,div,vol,T,t))
        bs_european_call_vega  = np.sqrt(T-t) * np.exp(-div*(T-t)) * S \
                                 * norm.pdf(self.d1(S,K,r,div,vol,T,t))
        bs_european_call_gamma = np.exp(-div*(T-t)) * norm.pdf(self.d1(S,K,r,div,vol,T,t))\
                                 / ( S * vol * np.sqrt(T-t) )
        bs_european_call_rho   = (T-t) * np.exp(-r*(T-t)) * K * self.N(self.d2(S,K,r,div,vol,T,t))
        return  bs_european_call_price, bs_european_call_delta,\
                bs_european_call_theta, bs_european_call_vega,\
                bs_european_call_gamma, bs_european_call_rho

    def Black_Scholes_European_Put(self, t, maturity_date, stock_price,\
        strike_price, interest_rate, dividend_yield, volatility):
        T = maturity_date; S = stock_price; K = strike_price
        r = interest_rate; div = dividend_yield; vol = volatility
        bs_european_put_price = -np.exp(-div*(T-t)) * S * self.N(-self.d1(S,K,r,div,vol,T,t))\
                                +np.exp(-r*(T-t)) * K * self.N(-self.d2(S,K,r,div,vol,T,t))
        bs_european_put_delta = -np.exp(-div*(T-t)) * self.N(-self.d1(S,K,r,div,vol,T,t))
        bs_european_put_theta = -div*np.exp(-div*(T-t))*S*self.N(-self.d1(S,K,r,div,vol,T,t))\
                                -(vol*np.exp(-div*(T-t))*S*norm.pdf(self.d1(S,K,r,div,vol,T,t)))\
                                /(2*np.sqrt(T-t))\
                                + r*np.exp(-r*(T-t))*K*self.N(-self.d2(S,K,r,div,vol,T,t))
        bs_european_put_vega  = np.sqrt(T-t) * np.exp(-div*(T-t)) * S \
                                 * norm.pdf(self.d1(S,K,r,div,vol,T,t))
        bs_european_put_gamma = (np.exp(-div*(T-t)) * norm.pdf(self.d1(S,K,r,div,vol,T,t)))\
                                 / ( S * vol * np.sqrt(T-t) )
        bs_european_put_rho   = -(T-t) * np.exp(-r*(T-t)) * K \
                                 * self.N(-self.d2(S,K,r,div,vol,T,t))
        return  bs_european_put_price, bs_european_put_delta,\
                bs_european_put_theta, bs_european_put_vega,\
                bs_european_put_gamma, bs_european_put_rho


    def Black_Scholes_Explicit_FD_EO(self, N, M, initial_condition, boundary_condition, \
                                     T, K, r, div, vol ):
        """
        :param N: number of interval in underlying (stock)
        :param M: number of interval in time [0,T]
        :param initial_condition: call or put
        :param boundary_condition:
        :param T: time to maturity
        :param S: underlying price at t=0
        :param K: strike price
        :param r: risk free interest rate
        :param div: dividend yield
        :param vol: underlying volatility
        :return: option price at t=0
        """

        # variable declaration
        import numpy as np
        import scipy.sparse
        smax = 100
        smin = 0.0
        ds = (smax - smin) / N
        dt = (T - 0.0) / M

        def low(i):
            return vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds - (r - div) * (
                                smin + (i - 1) * ds) * dt / ds / 2
        def down(i):
            return 1 - r * dt - vol ** 2 * (smin + (i - 1) * ds) ** 2 * dt / ds / ds
        def up(i):
            return vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds + (r - div) * (
                                        smin + (i - 1) * ds) * dt / ds / 2

        # Stability condition check
        if ((dt / ds / ds <= 0) or (dt / ds / ds >= 0.5)):
            # Courant-Friedrichs-Lewy condition
            print("warning: unstable")
            quit()

        # BC and IC set up
        if boundary_condition == "dirichlet_bc":
            print("dirichlet!")
            vCallmin = 0.0
            vCallmax = np.exp(-div * T) * smax - np.exp(-r * T) * K  # need to set up Smax
            vPutmin = np.exp(-r * T) * K - np.exp(-div * T) * 0.0
            vPutmax = 0.0
        elif boundary_condition == "neumann_bc":
            print("neumann!")
            self.stock_price = smin + ds * 2
            vCallmin = (2 * low(1) + down(1)) * 0.0 + (up(1) - low(1)) * self.European_Call_Option_Payoff()
            vPutmin = (2 * low(1) + down(1)) * (np.exp(-r * T) * K - np.exp(-div * T) * 0.0) + \
                      (up(1) - low(1)) * self.European_Put_Option_Payoff()
            self.stock_price = smin + ds * N
            vCallmax = (low(N + 1) - up(N + 1)) * self.European_Call_Option_Payoff() + \
                       (down(N + 1) + 2 * up(N + 1)) * (np.exp(-div * T) * smax - np.exp(-r * T) * K)
            vPutmax = (low(N + 1) - up(N + 1)) * self.European_Put_Option_Payoff() + \
                      (down(N + 1) + 2 * up(N + 1)) * (0.0)
        else:
            print("Warning: option boundary inputs are incorrect")
            print("Input either dirichlet_bc or neumann_bc")
        v1k = 0
        vN1k = 0
        if initial_condition == "ic_call":
            def u(x):
                self.stock_price = x
                return self.European_Call_Option_Payoff()
            v1k = vCallmin
            vN1k = vCallmax
            print("call: min ", v1k, ",max ", vN1k)
        elif initial_condition == "ic_put":
            def u(x):
                self.stock_price = x
                return self.European_Put_Option_Payoff()
            v1k = vPutmin
            vN1k = vPutmax
            print("put : min ", v1k, ",max ", vN1k)
        else:
            print("Warning: option IC inputs are incorrect")
            print("Input either ic_call or ic_put")
            quit()


        # Below runs only when Stability condition if okay

        # Define the weighting matrix
        def Wmat_Explicit(N):
            matrix_size = N
            row_number = matrix_size
            col_number = matrix_size
            sparse_matrix = scipy.sparse.coo_matrix((row_number, col_number))

        # Let the initial condition be
            for i in range(matrix_size):  # N is matrix-size. iteration by number of matrix_size
                if i == 0:
                    data = [down(i),up(i)]
                    row_index = [i, i]
                    col_index = [i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                elif i == matrix_size - 1:
                    data = [low(i), down(i)]
                    row_index = [i, i]
                    col_index = [i - 1, i]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                else:
                    data = [low(i),down(i),up(i)]
                    row_index = [i, i, i]
                    col_index = [i - 1, i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
            return sparse_matrix.toarray()

        # Define the weighting matrix
        A = Wmat_Explicit(N)
        # iteration
        U = np.asarray([u(i) for i in range(N)]).reshape(N, 1)  # initial condition
        d = np.zeros((N, 1))
        d[0] = low(2) * v1k
        d[N-1] = up(N) * vN1k
        for i in range(M):
            U = np.matmul(A, U) + d
        bs_explicit_fd_eo_price = U
        return bs_explicit_fd_eo_price


    def Black_Scholes_Implicit_FD_EO(self, N, M, initial_condition, boundary_condition, \
                                     T, K, r, div, vol ):
        """
        :param N: number of interval in underlying (stock)
        :param M: number of interval in time [0,T]
        :param initial_condition: call or put
        :param boundary_condition:
        :param T: time to maturity
        :param S: underlying price at t=0
        :param K: strike price
        :param r: risk free interest rate
        :param div: dividend yield
        :param vol: underlying volatility
        :return: option price at t=0
        """

        # variable declaration
        import numpy as np
        import scipy.sparse
        smax = 100
        smin = 0.0
        ds = (smax - smin) / N
        dt = (T - 0.0) / M

        def low(i):
            return -vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds + (r - div) * (
                    smin + (i - 1) * ds) * dt / ds / 2
        def down(i):
            return 1 + r * dt + vol ** 2 * (smin + (i - 1) * ds) ** 2 * dt / ds / ds
        def up(i):
            return -vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds - (r - div) * (
                    smin + (i - 1) * ds) * dt / ds / 2

        # BC and IC set up
        if boundary_condition == "dirichlet_bc":
            vCallmin = 0.0
            vCallmax = np.exp(-div * T) * smax - np.exp(-r * T) * K  # need to set up Smax
            vPutmin = np.exp(-r * T) * K - np.exp(-div * T) * 0.0
            vPutmax = 0.0
        elif boundary_condition == "neumann_bc":
            self.stock_price = smin + ds * 2
            vCallmin = (2 * low(1) + down(1)) * 0.0 + (up(1) - low(1)) * self.European_Call_Option_Payoff()
            vPutmin = (2 * low(1) + down(1)) * (np.exp(-r * T) * K - np.exp(-div * T) * 0.0) + \
                      (up(1) - low(1)) * self.European_Put_Option_Payoff()
            self.stock_price = smin + ds * N
            vCallmax = (low(N + 1) - up(N + 1)) * self.European_Call_Option_Payoff() + \
                       (down(N + 1) + 2 * up(N + 1)) * (np.exp(-div * T) * smax - np.exp(-r * T) * K)
            vPutmax = (low(N + 1) - up(N + 1)) * self.European_Put_Option_Payoff() + \
                      (down(N + 1) + 2 * up(N + 1)) * (0.0)
        else:
            print("Warning: option boundary inputs are incorrect")
            print("Input either dirichlet_bc or neumann_bc")

        if initial_condition == "ic_call":
            def u(x):
                self.stock_price = x
                return self.European_Call_Option_Payoff()
            v1k = vCallmin
            vN1k = vCallmax
        elif initial_condition == "ic_put":
            def u(x):
                self.stock_price = x
                return self.European_Put_Option_Payoff()
            v1k = vPutmin
            vN1k = vPutmax
        else:
            print("Warning: option IC inputs are incorrect")
            print("Input either ic_call or ic_put")
            quit()


        def Wmat_Implicit(N):
            matrix_size = N
            row_number = matrix_size
            col_number = matrix_size
            sparse_matrix = scipy.sparse.coo_matrix((row_number, col_number))

            # Let the initial condition be
            for i in range(matrix_size):  # N is matrix-size. iteration by number of matrix_size
                if i == 0:
                    data = [down(i), up(i)]
                    row_index = [i, i]
                    col_index = [i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                elif i == matrix_size - 1:
                    data = [low(i), down(i)]
                    row_index = [i, i]
                    col_index = [i - 1, i]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                else:
                    data = [low(i), down(i), up(i)]
                    row_index = [i, i, i]
                    col_index = [i - 1, i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
            return sparse_matrix.toarray()

        if boundary_condition == "dirichlet_bc":
            A = Wmat_Implicit(N)
            # iteration
            U = np.asarray([u(smin + ds * i) for i in range(N)]).reshape(N, 1)  # initial condition
            d = np.zeros((N, 1))
            d[0] = -low(2) * v1k
            d[N-1] = -up(N-1) * vN1k

            import scipy.linalg as sl
            for i in range(M):
                U = sl.lu_solve(sl.lu_factor(A), U + d)
        elif boundary_condition == "neumann_bc":
            A = Wmat_Implicit(N)
            A[0][0] += 2 * up(1)
            A[0][1] -= low(1)
            A[N-1][N - 2] -= up(N-1)
            A[N-1][N-1] += 2 * up(N-1)
            # iteration
            U = np.asarray([u(smin + ds * i) for i in range(N)]).reshape(N, 1)  # initial condition
            U[0] = v1k
            U[N-1] = vN1k

            import scipy.linalg as sl
            for i in range(M):
                U = sl.lu_solve(sl.lu_factor(A), U)

        bs_implicit_fd_eo_price = U
        return bs_implicit_fd_eo_price


    def Black_Scholes_Theta_FD_EO(self, N, M, initial_condition, boundary_condition, \
                                     T, K, r, div, vol, theta):
        """
        :param N: number of interval in underlying (stock)
        :param M: number of interval in time [0,T]
        :param initial_condition: call or put
        :param boundary_condition:
        :param T: time to maturity
        :param S: underlying price at t=0
        :param K: strike price
        :param r: risk free interest rate
        :param div: dividend yield
        :param vol: underlying volatility
        :param theta: [0,1]. 0: exp, 0.5: CN, 1: imp
        :return: option price at t=0
        """

        # variable declaration
        import numpy as np
        import scipy.sparse
        smax = 100
        smin = 0.0
        ds = (smax - smin) / N
        dt = (T - 0.0) / M

        def Elow(i):
            return vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds - (r - div) * (
                                smin + (i - 1) * ds) * dt / ds / 2
        def Edown(i):
            return 1 - r * dt - vol ** 2 * (smin + (i - 1) * ds) ** 2 * dt / ds / ds
        def Eup(i):
            return vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds + (r - div) * (
                                        smin + (i - 1) * ds) * dt / ds / 2

        def Ilow(i):
            return -vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds + (r - div) * (
                    smin + (i - 1) * ds) * dt / ds / 2
        def Idown(i):
            return 1 + r * dt + vol ** 2 * (smin + (i - 1) * ds) ** 2 * dt / ds / ds
        def Iup(i):
            return -vol ** 2 * (smin + (i - 1) * ds) ** 2 / 2 * dt / ds / ds - (r - div) * (
                    smin + (i - 1) * ds) * dt / ds / 2

        # BC and IC set up
        if boundary_condition == "dirichlet_bc":
            vCallmin = 0.0
            vCallmax = np.exp(-div * T) * smax - np.exp(-r * T) * K  # need to set up Smax
            vPutmin = np.exp(-r * T) * K - np.exp(-div * T) * 0.0
            vPutmax = 0.0
        elif boundary_condition == "neumann_bc":
            self.stock_price = smin + ds * 2
            vCallmin = (1-theta) * ((2 * Elow(1) + Edown(1)) * 0.0 + (Eup(1) - Elow(1)) * self.European_Call_Option_Payoff()) +\
                        theta * ((2 * Ilow(1) + Idown(1)) * 0.0 + (Iup(1) - Ilow(1)) * self.European_Call_Option_Payoff())
            vPutmin = (1-theta)* ((2 * Elow(1) + Edown(1)) * (np.exp(-r * T) * K - np.exp(-div * T) * 0.0) + \
                      (Eup(1) - Elow(1)) * self.European_Put_Option_Payoff()) + \
                      theta * ((2 * Ilow(1) + Idown(1)) * (np.exp(-r * T) * K - np.exp(-div * T) * 0.0) + \
                      (Iup(1) - Ilow(1)) * self.European_Put_Option_Payoff())
            self.stock_price = smin + ds * N
            vCallmax = (1-theta) * ((Elow(N + 1) - Eup(N + 1)) * self.European_Call_Option_Payoff() + \
                       (Edown(N + 1) + 2 * Eup(N + 1)) * (np.exp(-div * T) * smax - np.exp(-r * T) * K)) + \
                        theta * ((Ilow(N + 1) - Iup(N + 1)) * self.European_Call_Option_Payoff() + \
                       (Idown(N + 1) + 2 * Iup(N + 1)) * (np.exp(-div * T) * smax - np.exp(-r * T) * K))
            vPutmax = (1-theta) * ((Elow(N + 1) - Eup(N + 1)) * self.European_Put_Option_Payoff() + \
                      (Edown(N + 1) + 2 * Eup(N + 1)) * (0.0)) + \
                        theta * ((Ilow(N + 1) - Iup(N + 1)) * self.European_Put_Option_Payoff() + \
                      (Idown(N + 1) + 2 * Iup(N + 1)) * (0.0))

        else:
            print("Warning: option boundary inputs are incorrect")
            print("Input either dirichlet_bc or neumann_bc")

        if initial_condition == "ic_call":
            def u(x):
                self.stock_price = x
                return self.European_Call_Option_Payoff()
            v1k = vCallmin
            vN1k = vCallmax
        elif initial_condition == "ic_put":
            def u(x):
                self.stock_price = x
                return self.European_Put_Option_Payoff()
            v1k = vPutmin
            vN1k = vPutmax
        else:
            print("Warning: option IC inputs are incorrect")
            print("Input either ic_call or ic_put")
            quit()

        # Define the weighting matrix
        def Wmat_Explicit_theta(N, theta):
            matrix_size = N
            row_number = matrix_size
            col_number = matrix_size
            sparse_matrix = scipy.sparse.coo_matrix((row_number, col_number))

        # Let the initial condition be
            for i in range(matrix_size):  # N is matrix-size. iteration by number of matrix_size
                if i == 0:
                    data = [Edown(i),Eup(i)]
                    row_index = [i, i]
                    col_index = [i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                elif i == matrix_size - 1:
                    data = [Elow(i), Edown(i)]
                    row_index = [i, i]
                    col_index = [i - 1, i]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                else:
                    data = [Elow(i),Edown(i),Eup(i)]
                    row_index = [i, i, i]
                    col_index = [i - 1, i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
            sparse_matrix = (1-theta) * sparse_matrix + scipy.sparse.identity(matrix_size)
            return sparse_matrix.toarray()

        def Wmat_Implicit_theta(N, theta):
            matrix_size = N
            row_number = matrix_size
            col_number = matrix_size
            sparse_matrix = scipy.sparse.coo_matrix((row_number, col_number))

            # Let the initial condition be
            for i in range(matrix_size):  # N is matrix-size. iteration by number of matrix_size
                if i == 0:
                    data = [Idown(i), Iup(i)]
                    row_index = [i, i]
                    col_index = [i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                elif i == matrix_size - 1:
                    data = [Ilow(i), Idown(i)]
                    row_index = [i, i]
                    col_index = [i - 1, i]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
                else:
                    data = [Ilow(i), Idown(i), Iup(i)]
                    row_index = [i, i, i]
                    col_index = [i - 1, i, i + 1]
                    sparse_matrix = sparse_matrix + scipy.sparse.coo_matrix((data, (row_index, col_index)),
                                                                            shape=(row_number, col_number))
            sparse_matrix = theta * sparse_matrix + scipy.sparse.identity(matrix_size)
            return sparse_matrix.toarray()

        import scipy.linalg as sl
        # Define the weighting matrix
        AE_thetad_plus_I = Wmat_Explicit_theta(N, theta)
        AI_thetad_plus_I = Wmat_Implicit_theta(N, theta)
        # iteration
        V = np.asarray([u(i) for i in range(N)]).reshape(N, 1)  # initial condition
        d = np.zeros((N, 1))
        d[0] = (1-theta)*Elow(2)*v1k - theta*Ilow(2)*v1k
        d[N-1] = (1-theta)*Eup(N)*vN1k - theta*Iup(N)*vN1k
        for i in range(M):
            V = np.matmul(AE_thetad_plus_I, V) + d   # RHS p154
            V = sl.lu_solve(sl.lu_factor(AI_thetad_plus_I), V)  # LHS p154

        bs_theta_scheme_fd_eo_price = V
        return bs_theta_scheme_fd_eo_price

    def Black_Scholes_Global_RBF_EO(self, rbf_function, N, M, \
                                    initial_condition, boundary_condition, \
                                    T, K, r, vol, eps=1.5):
        # rbf_function: ga_rbf, mq_rbf, imq_rbf, iq_rbf
        # initial_condition: ic_call, ic_put
        # boundary_condition: dirichlet_bc
        import numpy as np
        from numpy.linalg import inv
        from math import exp

        smin = 0.001; smax = 100.0
        x = np.linspace(smin, smax, N)
        x_ = np.log(x)
        t = np.linspace(0.0, T, M)
        dt = T/M
        xi, x = np.meshgrid(x_, x_)
        self.stock_price = 0

        '''
        np.round(x)
        array([[  0.,   0.,   0.],
               [ 50.,  50.,  50.],
               [100., 100., 100.]])
        np.round(xi)
        array([[  0.,  50., 100.],
               [  0.,  50., 100.],
               [  0.,  50., 100.]])
        '''

        funcdict = {
            'ga_rbf': self.Gaussian_RBF,
            'mq_rbf': self.Multiquadratic_RBF,
            'imq_rbf': self.Inverse_Multiquadratic_RBF,
            'iq_rbf': self.Inverse_Quadratic_RBF
        }

        if rbf_function not in funcdict:
            print("please input legit rbf_functions.")
            quit()

        L, Lx, Lxx = funcdict[rbf_function](eps, x, xi)
        P = np.dot(-inv(L), ((r - 0.5 * vol ** 2) * Lx + 0.5 * vol ** 2 * Lxx - r * L))

        if boundary_condition == 'dirichlet_bc':   ## what'???
            def bcmin(init, t_i):
                if init == 'ic_call':
                    return 0
                elif init == 'ic_put':
                    return exp(-r * (T - t_i)) * K - smin
                else:
                    quit() # already error message was took care of above

            def bcmax(init, t_i):
                if init == 'ic_call':
                    return smax - exp(-r * (T - t_i)) * K
                elif init == 'ic_put':
                    return 0
                else:
                    quit()
        else:
            print("put only dirichlet_bc as a boundary_condition.")
            quit()

        # data init
        '''  v0 v1 ....vk......vM
        smin +-----------------+
             |
             |
             |
        smax +-----------------+
         tau=0(maturity)    tau=Tmax(t=0)
        '''
        # set the IC v1 at tau1
        v = np.zeros((N, 1))  # 0~N-1
        if initial_condition == 'ic_call':
            v[0] = bcmin(initial_condition, t[0])
            for i in range(1, N - 1):  # 1~N-2
                self.stock_price = exp(x_[i])  # x_: 1d vec
                v[i] = self.European_Call_Option_Payoff()
            v[N - 1] = bcmax(initial_condition, t[0])
        elif initial_condition == 'ic_put':
            v[0] = bcmin(initial_condition, t[0])
            for i in range(1, N - 1):
                self.stock_price = exp(x_[i])  # x_: 1d vec
                v[i] = self.European_Put_Option_Payoff()
            v[N - 1] = bcmax(initial_condition, t[0])
        else:
            print("put only dirichlet_bc as a boundary_condition.")
            quit()
        l = np.dot(inv(L), v)  # lambda 1 = inv(L) * v1 (6.2.33)
        for k in range(1, M):  # math(2:M)  python(1:M-1)
            l = np.dot(np.dot(inv(np.identity(N) - dt * P / 2), (np.identity(N) + dt * P / 2)), l)  # l 2
            v = np.dot(L, l)  # (6.2.35=13)  # v2
            v[0] = bcmin(initial_condition, t[k])  # (6.2.36)
            v[N - 1] = bcmax(initial_condition, t[k])  # (6.2.36)
            l = np.dot(inv(L), v)  # lambda 1 = inv(L) * v1 (6.2.33)

        bs_global_rbf_eo_price = v.T
        return bs_global_rbf_eo_price

    def Black_Scholes_RBF_FD_EO(self, rbf_function, N, M, \
                                    initial_condition, boundary_condition, \
                                    T, K, r, vol, eps=0.5):
        # rbf_function: ga_rbf, mq_rbf, imq_rbf, iq_rbf
        # initial_condition: ic_call, ic_put
        # boundary_condition: dirichlet_bc
        import numpy as np
        from numpy.linalg import inv
        from math import exp

        smax = 100
        smin = 0

        dt = T / N
        dx = (smax - smin) / M
        i = np.linspace(1, M + 1, M + 1)
        s_value = smin + dx * i
        s_value = s_value[1:M]
        x = np.log(s_value)
        xi = np.reshape(x, (M - 1, 1))

        funcdict = {
            'ga_rbf': self.Gaussian_RBF,
            'mq_rbf': self.Multiquadratic_RBF,
            'imq_rbf': self.Inverse_Multiquadratic_RBF,
            'iq_rbf': self.Inverse_Quadratic_RBF
        }

        if rbf_function not in funcdict:
            print("please input legit rbf_functions.")
            quit()

        L, Lx, Lxx = funcdict[rbf_function](eps, x, xi)
        P = np.dot(-inv(L), ((r - 0.5 * vol ** 2) * Lx + 0.5 * vol ** 2 * Lxx - r * L))

        if boundary_condition == 'dirichlet_bc':   ## what'???
            def bcmin(init, t_i):
                if init == 'ic_call':
                    return 0
                elif init == 'ic_put':
                    return np.exp(-r * (T-t_i)) * K - s_value[0]
                else:
                    quit() # already error message was took care of above

            def bcmax(init, t_i):
                if init == 'ic_call':
                    return s_value[-1] - np.exp(-r * (T-t_i)) * K
                elif init == 'ic_put':
                    return 0
                else:
                    quit()
        else:
            print("put only dirichlet_bc as a boundary_condition.")
            quit()

        # set the IC v1 at tau1

        W = np.zeros((M - 1, M - 1))
        for i in range(M - 1):
            linear_operator = (r - 0.5 * (vol ** 2)) * Lx[:, i] + (0.5 * (vol ** 2)) * Lxx[:, i] - r * L[:, i]
            W[:, i] = np.linalg.solve(L, linear_operator)
        W = np.matrix.transpose(W)

        if initial_condition == 'ic_call':
            v = np.maximum(np.exp(x) - K, 0)
        else:
            v = np.maximum(K - np.exp(x), 0)
        v[0] = bcmin(initial_condition, 0)
        v[-1] = bcmax(initial_condition, 0)

        tmp = []
        for t in range(N):
            v = np.linalg.solve((np.identity(np.size(W, 1)) - 0.5 * dt * W),
                                ((np.identity(np.size(W, 1)) + 0.5 * dt * W) @ v))
            v[0] = bcmin(initial_condition, t * dt)
            v[-1] = bcmax(initial_condition, t * dt)
            tmp.append(v)

        bs_rbf_fd_eo_price = tmp
        return bs_rbf_fd_eo_price

    def Geometric_Brownian_Trajectory(self, init_time, maturity, time_number, S0, simul_number,\
                                      mu, sigma):
        '''

        :param init_time: t0. when is initial time. usually set 0
        :param maturity: T. when is end of period (year := 1, month := 1/12)
        :param time_number: n, how many intervals are there
        :param S0: current value of underlying (stock)
        :param simul_number: N. how many trajectories will we generate
        :param mu: mean of underlying return (yearly)
        :param sigma: volatility
        :return: array of numpy arrays(1D vector)
        '''
        import scipy as sp
        import numpy as np
        from math import exp, sqrt
        t0 = init_time
        T = maturity
        n = time_number
        N = simul_number
        dt = (T - t0) / n
        trajectories = np.array(np.zeros(n))
        for i in range(N):
            S = S0
            path = [S0]
            for j in range(n - 1):
                z = sp.random.standard_normal()
                S = S * exp((mu - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z)
                path.append(S)
            trajectories = np.vstack((trajectories, np.asarray(path)))
        return trajectories[1:]

    def Geometric_Brownian_Motion_Jump(self, init_time, maturity, time_number, S0, simul_number,\
                                      mu, sigma, div=0.01, poi_lambda=1.0, jump_a=0.0005, jump_b=0.0015):
        '''
        :param init_time: t0. when is initial time. usually set 0
        :param maturity: T. when is end of period (year := 1, month := 1/12)
        :param time_number: n, how many intervals are there
        :param S0: current value of underlying (stock) :: it will be logged
        :param simul_number: N. how many trajectories will we generate
        :param mu: mean of underlying return (yearly)
        :param sigma: volatility
        :param div: dividend yield, default 0.01
        :param poi_lambda: default 1.0
        :param jump_a: default 0.0005
        :param jump_b: default 0.0015
        :return:
        '''
        import scipy as sp
        import numpy as np
        from math import sqrt, log
        t0 = init_time
        T = maturity
        n = time_number
        N = simul_number
        dt = (T - t0) / n
        X0 = log(S0)
        trajectories = np.array(np.zeros(n))
        a = jump_a
        b = jump_b
        for i in range(N):
            X = X0
            path = [X]
            for j in range(n - 1):
                z1 = sp.random.standard_normal()
                poi = sp.random.poisson(lam=poi_lambda * dt)
                if poi == 0:
                    M = 0
                else:
                    z2 = sp.random.standard_normal()
                    M = a * n + b * sqrt(n) * z2
                X = X + ((mu - div) - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z1 + M
                path.append(X)
            trajectories = np.vstack((trajectories, np.asarray(path)))
        paths = np.exp(trajectories[1:])
        return paths
    def Arithmetic_Average_Price_Asian_Call(self, init_time, maturity, time_number, S0, simul_number,\
                                      mu, sigma, div=0.01, poi_lambda=1.0, jump_a=0.0005, jump_b=0.0015):
        '''

        :param init_time: t0
        :param maturity:  T
        :param time_number: 252 for a year
        :param S0:
        :param simul_number: path number ie. 1k, 10k, ...
        :param mu:
        :param sigma:
        :param div:
        :param poi_lambda:
        :return:
        '''
        from math import exp
        paths = self.Geometric_Brownian_Motion_Jump(init_time, maturity, time_number, S0, simul_number, \
                                       mu, sigma, div, poi_lambda, jump_a, jump_b)
        C0 = []
        for i in range(paths.shape[0]):
            Sa = np.average(paths[i])
            CT_i = max(Sa - self.strike_price, 0)
            C0_i = exp(-mu * (maturity - init_time)) * CT_i
            C0.append(C0_i)  # pv vector
        ArithAsianCall = np.average(np.asarray(C0))
        ArithAsianCallSTD = np.std(np.asarray(C0))
        return ArithAsianCall, ArithAsianCallSTD, np.asarray(C0)

    def Geometric_Average_Price_Asian_Call(self, init_time, maturity, time_number, S0, simul_number,\
                                      mu, sigma, div=0.01, poi_lambda=1.0, jump_a=0.0005, jump_b=0.0015):
        from math import exp
        paths = self.Geometric_Brownian_Motion_Jump(init_time, maturity, time_number, S0, simul_number, \
                                       mu, sigma, div, poi_lambda, jump_a, jump_b)
        C0 = []
        for i in range(paths.shape[0]):
            Sg = exp(np.sum(np.log(paths[i])) / paths.shape[1])  # exp(log(X1 X2 ... Xn)) ** (1/n)
            CT_i = max(Sg - self.strike_price, 0)
            C0_i = exp(-mu*(maturity-init_time)) * CT_i
            C0.append(C0_i)  # pv vector
        GeomAsianCall = np.average(np.asarray(C0))
        GeomAsianCallSTD = np.std(np.asarray(C0))
        return GeomAsianCall, GeomAsianCallSTD, np.asarray(C0)

    def BS_Geometric_Average_Price_Asian_Call(self, init_time, maturity, S0, mu, sigma, div=0.01):
        from math import sqrt, log, exp
        t0 = init_time
        T = maturity
        sigtilda = sigma/sqrt(3)
        b = 0.5*(mu + div + sigma**2 / 6)
        d1 = (log(S0/self.strike_price) + (mu-b+.5 * sigtilda**2)*(T-t0)) / (sigtilda * sqrt(T-t0)) # cannot use self.d1()
        d2 = d1 - sigtilda*sqrt(T-t0)
        return S0*exp(-b*(T-t0))*self.N(d1)-self.strike_price*exp(-mu*(T-t0))*self.N(d2)

    def Control_Variates_Arithmetic_Average_Asian_Call(self, init_time, maturity, time_number, S0, simul_number,\
                                      mu, sigma, div=0.01, poi_lambda=1.0, jump_a=0.0005, jump_b=0.0015):
        EX = self.BS_Geometric_Average_Price_Asian_Call(init_time, maturity, S0, mu, sigma, div)
        withoutCVprice, withoutCVstd, withoutCVpi = \
            self.Arithmetic_Average_Price_Asian_Call(init_time, maturity, time_number, S0, simul_number, \
                                                     mu, sigma, div, poi_lambda, jump_a, jump_b)
        Gbar, Gstd, G0 = \
            self.Geometric_Average_Price_Asian_Call(init_time, maturity, time_number, S0, simul_number,\
                                                     mu, sigma, div, poi_lambda, jump_a, jump_b)
        b = np.dot(G0 - Gbar, withoutCVpi - withoutCVprice) / (Gstd**2 * simul_number)
        #Gstd_ = np.sqrt( np.dot(G0 - Gbar, G0 - Gbar) / simul_number )
        withCVpi = withoutCVpi - b * (G0 - EX)
        withCVprice = np.average(withCVpi)
        withCVstd = np.std(withCVpi)
        return withCVprice, withCVstd, withoutCVprice, withoutCVstd, withCVpi, withoutCVpi, b

        # Black-Scholes characteristic function for Fourier Transform
    def BS_phi(self, v, x0, T, r, sigma):  # [Ng(2005)] p32
        return np.exp(((x0 / T + r - 0.5 * sigma ** 2) * 1j * v - 0.5 * sigma ** 2 * v ** 2) * T)

    def Option_Pricing_FFT_EO(self, S0, K, T, r, sigma):
        '''
        (Call) Option pricing by FFT
        :param S0: current price of underlying
        :param K: strike price
        :param T: time-to-maturity
        :param r: risk free rate
        :param sigma: volatility
        :return: Call value
        '''
        import numpy as np
        from numpy.fft import fft
        # Each parenthesis (..) in below comment denotes the equation number in Carr & Madan (1998)

        # Setting parameters
        k = np.log(K)    # implied in p68 L3~4
        x0 = np.log(S0)  # implied in p68 L3~4
        N = 4096  # p69 L-3, 2^k size
        lamb = 1 / 200.  # lambda, for 200 strike levels
        b = .5 * N * lamb  # (20)
        eta = 2 * np.pi / (N * lamb)  # (23)
        u = np.arange(1, N+1, 1)  # possible  u = 1, ..., N  for (19)
        vj = eta * (u - 1)  # p67 L-2

        # Gets the Phi(vj), deviding the case into ITM and OTM, each applying the part introduced in the paper
        if S0 > K:  # ITM
            alpha = 1.5  # p70 L2
            v = vj - (alpha + 1) * 1j  # (6)
            Phi = np.exp(-r * T) * (self.BS_phi(v, x0, T, r, sigma) /
                    (alpha ** 2 + alpha - vj ** 2 + 1j * (2 * alpha + 1) * vj))  # (6)
        else:  # OTM (15)
            alpha = 1.1  # p70 L2
            v = (vj - 1j * alpha) - 1j
            Phi1 = np.exp(-r * T) * (1 / (1 + 1j * (vj - 1j * alpha)) -
                    np.exp(r * T) / (1j * (vj - 1j * alpha)) - self.BS_phi(v, x0, T, r, sigma) /
                    ((vj - 1j * alpha) ** 2 - 1j * (vj - 1j * alpha)))  # (14)
            v = (vj + 1j * alpha) - 1j
            Phi2 = np.exp(-r * T) * (1 / (1 + 1j * (vj + 1j * alpha)) -
                    np.exp(r * T) / (1j * (vj + 1j * alpha)) - self.BS_phi(v, x0, T, r, sigma) /
                    ((vj + 1j * alpha) ** 2 - 1j * (vj + 1j * alpha)))  # (14)
        krondelta = np.zeros(N, dtype=np.float) # Kronecker delta step 1
        krondelta[0] = 1  # Kronecker delta step 2
        j = np.arange(1, N + 1, 1)  # j = 1:N
        Simpson = (3 + (-1) ** j - krondelta) / 3  # Simpson's rule weightings and the restriction (23)
        if S0 > K: # ITM FFT implementation
            FFTfunction = np.exp(1j * b * vj) * Phi * eta * Simpson  # (24), Call price formula
            fftres = (fft(FFTfunction)).real # (24), FFT using numpy.fft
            Call = np.exp(-alpha * k) / np.pi * fftres # (24), adjustment
        else: # OTM
            FFTfunction = np.exp(1j * b * vj) * (Phi1 - Phi2) * 0.5 * eta * Simpson  # (24)' p68 L-2
            fftres = (fft(FFTfunction)).real # (24), FFT using numpy.fft
            Call = fftres / (np.sinh(alpha * k) * np.pi)  # p68 L-2
        CallValue = Call[int(k/lamb + .5*N)]
        return CallValue

    def BS_call_integrate(self, S0, K, T, r, sigma):  # BS by integration, Carr and Madan (1999) p.62~63
        '''
        Carr and Madan (1999) p.62~63
        :param S0: underlying, current
        :param K: strike
        :param T: time-to-maturity
        :param r: risk free rate
        :param sigma: vol
        :return: call value based on integration
        '''
        import numpy as np
        def BS_compo(S0, K, T, r, sigma, compo):
            # changing argument in the phi makes PI1, PI2
            from scipy.integrate import quad
            if compo == "call": # Risk Neutral Prob. of finishing in-the-money
                return 1 / 2 + 1 / np.pi * \
                       quad(lambda u: (np.exp(-1j * u * np.log(K)) * self.BS_phi(u, np.log(S0), T, r, sigma)
                                       / (1j * u)).real, 0, 200)[0]
            elif compo == "delta": # delta of option
                return 1 / 2 + 1 / np.pi * \
                       quad(lambda u: (np.exp(-1j * u * np.log(K)) * self.BS_phi(u-1j, np.log(S0), T, r, sigma)
                                       / (1j * u * self.BS_phi(-1j, np.log(S0), T, r, sigma))).real, 0, 200)[0]
            else:
                print("error")
                quit()
        PI2 = BS_compo(S0, K, T, r, sigma, "call")  # Risk Neutral Prob. of finishing in-the-money
        PI1 = BS_compo(S0, K, T, r, sigma, "delta") # delta of option
        call = S0*PI1-K*np.exp(-r*T)*PI2 # p63
        return call

        # VG characteristic function for Fourier Transform, from (26)
    def VG_phi(self, u, x0, T, r, sigma, nu, theta):
        w = (1 / nu) * np.log(1 - theta * nu - .5 * sigma ** 2 * nu)
        return np.exp(x0 + (r + w) * T) * (1 - 1j * theta * nu * u + .5 * sigma ** 2 * u ** 2 * nu) ** (-T / nu)

    def VG_Option_Pricing_FFT_EO(self, S0, K, T, r, sigma, nu, theta):
        '''
        (Call) Option pricing by FFT, based on (1998)
        :param S0: current price of underlying
        :param K: strike price
        :param T: time-to-maturity
        :param r: risk free rate
        :param sigma: volatility
        :return: Call value
        '''
        import numpy as np
        from numpy.fft import fft

        # Setting parameters
        k = np.log(K)    # implied in p68 L3~4
        x0 = np.log(S0)  # implied in p68 L3~4
        N = 4096  # p69 L-3, 2^k size
        lamb = 1 / 200.  # lambda, for 200 strike levels
        b = .5 * N * lamb  # (20)
        eta = 2 * np.pi / (N * lamb)  # (23)
        u = np.arange(1, N+1, 1)  # possible  u = 1, ..., N  for (19)
        vj = eta * (u - 1)  # p67 L-2
        if S0 > K:  # ITM
            alpha = 1.5  # p70 L2
            v = vj - (alpha + 1) * 1j  # (6)
            Phi = np.exp(-r * T) * (self.VG_phi(v, x0, T, r, sigma, nu, theta) /
                    (alpha ** 2 + alpha - vj ** 2 + 1j * (2 * alpha + 1) * vj))  # (6)
        else:  # OTM (15)
            alpha = 1.1  # p70 L2
            v = (vj - 1j * alpha) - 1j
            Phi1 = np.exp(-r * T) * (1 / (1 + 1j * (vj - 1j * alpha)) -
                    np.exp(r * T) / (1j * (vj - 1j * alpha)) - self.VG_phi(v, x0, T, r, sigma, nu, theta) /
                    ((vj - 1j * alpha) ** 2 - 1j * (vj - 1j * alpha)))  # (14)
            v = (vj + 1j * alpha) - 1j
            Phi2 = np.exp(-r * T) * (1 / (1 + 1j * (vj + 1j * alpha)) -
                    np.exp(r * T) / (1j * (vj + 1j * alpha)) - self.VG_phi(v, x0, T, r, sigma, nu, theta) /
                    ((vj + 1j * alpha) ** 2 - 1j * (vj + 1j * alpha)))  # (14)
        krondelta = np.zeros(N, dtype=np.float)
        krondelta[0] = 1  # Kronecker delta
        j = np.arange(1, N + 1, 1)  # j = 1:N
        Simpson = (3 + (-1) ** j - krondelta) / 3  # (24) latter part
        if S0 > K: # ITM
            FFTfunction = np.exp(1j * b * vj) * Phi * eta * Simpson  # (24)
            fftres = (fft(FFTfunction)).real
            Call = np.exp(-alpha * k) / np.pi * fftres
        else:
            FFTfunction = np.exp(1j * b * vj) * (Phi1 - Phi2) * 0.5 * eta * Simpson  # (24)' p68 L-2
            fftres = (fft(FFTfunction)).real
            Call = fftres / (np.sinh(alpha * k) * np.pi)  # p68 L-2
        CallValue = Call[int(k/lamb + .5*N)]
        return CallValue

    def VG_call_integrate(self, S0, K, T, r, sigma, nu, theta):
        '''
        Carr and Madan (1999) p.62~63, based on (1998)
        :param S0: underlying, current
        :param K: strike
        :param T: time-to-maturity
        :param r: risk free rate
        :param sigma: vol
        :return: call value based on integration
        '''
        import numpy as np
        def VG_compo(S0, K, T, r, sigma, compo, nu, theta):
            # changing argument in the phi makes PI1, PI2
            from scipy.integrate import quad
            if compo == "call": # Risk Neutral Prob. of finishing in-the-money
                return 1 / 2 + 1 / np.pi * \
                       quad(lambda u: (np.exp(-1j * u * np.log(K)) * self.VG_phi(u, np.log(S0), T, r, sigma, nu, theta)
                                       / (1j * u)).real, 0, 200)[0]
            elif compo == "delta": # delta of option
                return 1 / 2 + 1 / np.pi * \
                       quad(lambda u: (np.exp(-1j * u * np.log(K)) * self.VG_phi(u-1j, np.log(S0), T, r, sigma, nu, theta)
                                       / (1j * u * self.VG_phi(-1j, np.log(S0), T, r, sigma, nu, theta))).real, 0, 200)[0]
            else:
                print("error")
                quit()
        PI2 = VG_compo(S0, K, T, r, sigma, "call", nu, theta)  # Risk Neutral Prob. of finishing in-the-money
        PI1 = VG_compo(S0, K, T, r, sigma, "delta", nu, theta) # delta of option
        call = S0*PI1-K*np.exp(-r*T)*PI2 # p63
        return call