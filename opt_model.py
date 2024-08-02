from typing import List
import numpy as np
from scipy.optimize import minimize, fsolve
from utils import calc_au_p_fsolve, calc_alpha_asym, calc_access_delay_u, calc_conf, calc_mu_S

def calc_us_p_fsolve(n1: int, lambda1: float, n2: int, lambda2: float, W_1: int, K_1: int, W_2: int, K_2: int,  tt: float, tf: float):
    def psf(p, n_u, n_s, W_s, K_s, lambda_u):
        p_ans = np.zeros(2)
        alpha = calc_alpha_asym(tt, tf, n_s, p[0], n_u, p[1])
        p_ans[0] = p[0] - (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1))) ) ** n_s *\
                        (1 - lambda_u / (alpha * tt * p[0])) ** (n_u-1)
        p_ans[1] = p[1] - (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1))) ) ** (n_s-1) *\
                        (1 - lambda_u  / (alpha * tt * p[0])) ** n_u
        return p_ans
    p_us = fsolve(psf, [0.9, 0.9], args = (n1, n2, W_2, K_2, lambda1))
    err1 = np.sqrt(np.sum(np.array(psf(p_us, n2, n1, W_2, K_2, lambda1))**2))
    us = True
    if err1 > 1e-5:
        us = False
    return p_us, us
class opt_Model:
    def __init__(self, nmld: int, lambda_mld: float, nsld: List[int], lambda_sld: List[float], tt: List[float], tf: List[float], Wmld: List[int] = [16, 16], Kmld: List[int] = [6, 6], Wsld: List[int] = [16, 16], Ksld: List[int] = [6, 6]) -> None:
        '''
        lambda_mld: per slot 
        lambda_sld: per slot
        '''
        self.lambda_mld = lambda_mld
        self.lambda_sld = lambda_sld
        self.nmld = nmld
        self.nsld = nsld
        self.tt = tt
        self.tf = tf
        self.Wmld = Wmld
        self.Kmld = Kmld
        self.Wsld = Wsld
        self.Ksld = Ksld
        self.nlink = len(Wmld)
    
    def calc_tt_tf(self, Header, Payload, SIFS, ACK, DIFS, sigma, slot):
        '''
        sigma: propagation delay
        '''
        tt = Header + Payload + SIFS + ACK + DIFS + 2 * sigma
        tf = Header + Payload + DIFS + sigma
        return tt / slot, tf / slot
    
    
    def delay_of_beta(self, beta):
        return beta * self.delay(beta, 0) + (1 - beta) * self.delay(1-beta, 1)
    
    def delay(self, beta, linkid):
        lambda1 = beta * self.lambda_mld
        nsld = self.nsld[linkid]
        lambda2 = self.lambda_sld[linkid]
        tt = self.tt[linkid]
        tf = self.tf[linkid]
        lambda1 = lambda1 * tt
        lambda2 = lambda2 * tt
        p_uu, flag = calc_au_p_fsolve(self.nmld, nsld, lambda1, lambda2, tt, tf)
        if flag:
            self.alpha = calc_alpha_asym(tt, tf, self.nmld, p_uu[0], nsld, p_uu[1])
            qd1, ad1, ads1 = calc_access_delay_u(p_uu[0], self.alpha, tt, tf, self.Wmld[linkid], self.Kmld[linkid], lambda1)
            # qd2, ad2, ads2= calc_access_delay_u(p_uu[1], self.alpha, tt, tf, self.Wsld[linkid], self.Ksld[linkid], lambda2)
            self.p1, self.p2 = p_uu[0], p_uu[1]
            self.state = "UU"
            self.throughput1_Mbps = lambda1 * self.nmld / tt
            self.throughput2_Mbps = lambda2 * nsld / tt
            return ad1 + qd1
        else:
            p_us, us = calc_us_p_fsolve(self.nmld, lambda1, nsld, lambda2, self.Wmld[linkid], self.Kmld[linkid], self.Wsld[linkid], self.Ksld[linkid], tt, tf)
            self.alpha_us = calc_alpha_asym(tt, tf, self.nmld, p_us[0], nsld, p_us[1])
            cf_us = calc_conf(p_us[0], p_us[1], lambda1, lambda2, self.Wmld[linkid], self.Kmld[linkid], self.Wsld[linkid], self.Ksld[linkid], tt, tf, self.alpha_us, 1)
            if us:
                self.alpha = self.alpha_us
                pi_ts_us1 = calc_mu_S(p_us[0], tt, tf, self.Wmld[linkid], self.Kmld[linkid], self.alpha)
                pi_ts_us2 = calc_mu_S(p_us[1], tt, tf, self.Wsld[linkid], self.Ksld[linkid], self.alpha)
                self.throughput_1 = lambda1 * self.nmld
                self.throughput_2 = min(lambda2, pi_ts_us2) * nsld
                qd1, ad1, ads1 = calc_access_delay_u(p_us[0], self.alpha, tt, tf, self.Wmld[linkid], self.Kmld[linkid], lambda1)
                # qd2, ad2, ads2 = calc_access_delay_s(p_us[1], self.alpha, tt, tf, W_2, K_2, lambda2, pi_ts_us2)
                self.p1 = p_us[0]
                self.p2 = p_us[1]
                self.state = "US" 
                return qd1 + ad1
            else:
                return 1e5
            
    def opt_delay(self):
        def e2e_delay(beta):
            return beta * self.delay(beta, 0) + (1 - beta) * self.delay(1 - beta, 1)
        res = minimize(e2e_delay, 0.6, bounds=[(0, 1)])
        # return 1,2
        return res.x, res.fun
    
    def opt_delay_var_W(self, w_range):
        wwx, wwy = np.meshgrid(w_range, w_range)
        best_beta = []
        best_delay = []
        for w in zip(wwx, wwy):
            self.Wmld = w
            x, fun = self.opt_delay()
            best_beta.append(x)
            best_delay.append(fun)
        return best_beta, best_delay
    
    
if __name__ == "__main__":
    model = opt_Model(
        nmld = 1,
        lambda_mld = 0.002,
        nsld = [10, 10],
        lambda_sld = [0.001, 0.002],
        tt = [32, 32],
        tf = [27, 27]
    )
    x, y = model.opt_delay()
    print(x, y)
    w_range = [1, 2, 8]
    best_beta, best_delay = model.opt_delay_var_W(w_range)
    print("W_range: ", w_range)
    print("best_beta: ", best_beta)
    print("best_delay: ", best_delay)
    # wwx, wwy = np.meshgrid(range(1, 8), range(1, 8))
    # for i, j in zip(wwx.ravel(), wwy.ravel()):
    #     print(i, j)
    