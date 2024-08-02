from utils import calc_mu_S, calc_au_p_fsolve, calc_as_p_fsolve, calc_ps_p_fsolve,  calc_ps_p_formula, calc_uu_p_formula, calc_access_delay_s, calc_access_delay_u, calc_conf, calc_alpha_sym, calc_alpha_base, calc_alpha_asym, calc_mu_U
import numpy as np
class HOL_Model:
    def __init__(self, n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf) -> None:
        self.tt = tt
        self.tf = tf
        self.n1 = n1
        self.n2 = n2
        p_uu, flag = calc_au_p_fsolve(n1, n2, lambda1, lambda2, tt, tf)
        if flag:
            self.alpha = calc_alpha_asym(tt, tf, n1, p_uu[0], n2, p_uu[1])
            qd1, ad1, ads1 = calc_access_delay_u(p_uu[0], self.alpha, tt, tf, W_1, K_1, lambda1)
            qd2, ad2, ads2= calc_access_delay_u(p_uu[1], self.alpha, tt, tf, W_2, K_2, lambda2)
            self.queuing_delay_1 = qd1
            self.queuing_delay_2 = qd2
            self.access_delay_1 = ad1
            self.access_delay_2 = ad2
            self.access_delay_sec_1 = ads1
            self.access_delay_sec_2 = ads2
            self.p1, self.p2 = p_uu[0], p_uu[1]
            self.state = "UU"
            self.throughput_1 = lambda1 * n1
            self.throughput_2 = lambda2 * n2
            mu_S = calc_mu_S(p_uu[0], tt, tf, W_1, K_1, self.alpha)
            mu_U = calc_mu_U(p_uu[0], tt, tf, W_1, K_1, lambda1, self.alpha)
            print("mu: ",mu_S, mu_U)
        else:
            p_as, flag_ss = calc_as_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
            alpha_ss = calc_alpha_asym(tt, tf, n1, p_as[0], n2, p_as[1])
            cf_ss = calc_conf(p_as[0], p_as[1], lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, alpha_ss, 3)
            pi_ts_1 = calc_mu_S(p_as[0], tt, tf, W_1, K_1, alpha_ss)
            pi_ts_2 = calc_mu_S(p_as[1], tt, tf, W_2, K_2, alpha_ss)
            mu_u = calc_mu_U(p_as[0], tt, tf, W_1, K_1, lambda1, alpha_ss)
            print("mu_as:", pi_ts_1, pi_ts_2, mu_u)
            p_us, p_su, us, su = calc_ps_p_fsolve(n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf)
            # print(p_us, p_su, us, su)
            cf_us, cf_su = 0, 0
            self.alpha_us = calc_alpha_asym(tt, tf, n1, p_us[0], n2, p_us[1])
            cf_us = calc_conf(p_us[0], p_us[1], lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, self.alpha_us, 1)
            self.alpha_su = calc_alpha_asym(tt, tf, n1, p_su[0], n2, p_su[1])
            cf_su = calc_conf(p_su[0], p_su[1], lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, self.alpha_su, 2)
            cf_list = [cf_us, cf_su, cf_ss]
            assert np.max(cf_list) != 0
            best_idx = np.argmax(cf_list)
            if best_idx == 0: # US
                self.alpha = self.alpha_us
                pi_ts_us1 = calc_mu_S(p_us[0], tt, tf, W_1, K_1, self.alpha)
                pi_ts_us2 = calc_mu_S(p_us[1], tt, tf, W_2, K_2, self.alpha)
                self.throughput_1 = lambda1 * n1
                self.throughput_2 = min(lambda2, pi_ts_us2) * n2
                qd1, ad1, ads1 = calc_access_delay_u(p_us[0], self.alpha, tt, tf, W_1, K_1, lambda1)
                qd2, ad2, ads2 = calc_access_delay_s(p_us[1], self.alpha, tt, tf, W_2, K_2, lambda2, pi_ts_us2)
                self.access_delay_sec_1 = ads1
                self.access_delay_sec_2 = ads2
                self.access_delay_1 = ad1
                self.access_delay_2 = ad2
                qd1 = qd1 if qd1 > 0 else 1e5
                qd2 = qd2 if qd2 > 0 else 1e5
                self.queuing_delay_1 = qd1
                self.queuing_delay_2 = qd2
                self.p1 = p_us[0]
                self.p2 = p_us[1]
                self.state = "US"
            elif best_idx == 1: # SU
                self.alpha = self.alpha_su
                pi_ts_su1 = calc_mu_S(p_su[0], tt, tf, W_1, K_1, self.alpha)
                pi_ts_su2 = calc_mu_S(p_su[1], tt, tf, W_2, K_2, self.alpha)
                self.throughput_1 = min(lambda1, pi_ts_su1) * n1
                self.throughput_2 = lambda2 * n2
                qd1, ad1, ads1 = calc_access_delay_s(p_su[0], self.alpha, tt, tf, W_1, K_1, lambda1, pi_ts_su1)
                qd2, ad2, ads2 = calc_access_delay_u(p_su[1], self.alpha, tt, tf, W_2, K_2, lambda2)
                self.access_delay_sec_1 = ads1
                self.access_delay_sec_2 = ads2
                self.access_delay_1 = ad1
                self.access_delay_2 = ad2
                qd1 = qd1 if qd1 > 0 else 1e5
                qd2 = qd2 if qd2 > 0 else 1e5
                self.queuing_delay_1 = qd1
                self.queuing_delay_2 = qd2
                self.p1 = p_su[0]
                self.p2 = p_su[1]
                self.state = "SU"
            elif best_idx == 2: # SS
                self.throughput_1 = min(lambda1, pi_ts_1) * n1
                self.throughput_2 = min(lambda2, pi_ts_2) * n2
                print("SS: ", W_1, pi_ts_1, lambda1, pi_ts_2, lambda2)
                self.alpha = alpha_ss
                qd1, ad1, ads1 = calc_access_delay_s(p_as[0], self.alpha, tt, tf, W_1, K_1, lambda1, pi_ts_1)
                qd2, ad2, ads2 = calc_access_delay_s(p_as[1], self.alpha, tt, tf, W_2, K_2, lambda2, pi_ts_2)
                self.access_delay_sec_1 = ads1
                self.access_delay_sec_2 = ads2
                self.access_delay_1 = ad1
                self.access_delay_2 = ad2
                qd1 = qd1 if qd1 > 0 else 1e5
                qd2 = qd2 if qd2 > 0 else 1e5
                self.queuing_delay_1 = qd1
                self.queuing_delay_2 = qd2
                self.p1 = p_as[0]
                self.p2 = p_as[1]
                self.state = "SS"
        print(self.state, self.p)
    @property
    def p(self):
        return self.p1, self.p2
    
    @property
    def throughput(self): # per slot 
        return self.throughput_1/self.tt, self.throughput_2/self.tt 
    
    @property
    def throughput_Mbps(self): # per slot Mbps
        return self.throughput_1/self.tt * 1500 * 8 / 9, self.throughput_2/self.tt * 1500 * 8 / 9
    
    @property
    def access_delay_ms(self): # ms
        return self.access_delay_1 * 9 * 1e-3, self.access_delay_2 * 9 * 1e-3
    
    @property
    def access_delay(self): #
        return self.access_delay_1 , self.access_delay_2 
    
    @property
    def access_delay_sec(self): #
        return self.access_delay_sec_1 , self.access_delay_sec_2 
    
    @property
    def queuing_delay_ms(self):
        return self.queuing_delay_1 * 9 * 1e-3, self.queuing_delay_2 * 9 * 1e-3
    
    @property
    def queuing_delay(self):
        return self.queuing_delay_1, self.queuing_delay_2
    
    @property
    def e2e_delay(self):
        return self.queuing_delay_1 + self.access_delay_1, self.queuing_delay_2 + self.access_delay_2 
    
    @property
    def e2e_delay_ms(self):
        return (self.queuing_delay_1 + self.access_delay_1) * 9 * 1e-3, (self.queuing_delay_2 + self.access_delay_2) * 9 * 1e-3

    
if __name__ == "__main__":
    n1 = 20  
    n2 = 20  
    tau_T = 32
    tau_F = 27
    
    lam1 = 0.9 / n1 / tau_T
    lam2 = 0.7 / n2 / tau_T
    model = HOL_Model(
        n1 = n1/2,
        n2 = n1/2,
        lambda1 = tau_T * lam1,
        lambda2 = tau_T * lam1,
        W_1 = 16,
        W_2 = 16,
        K_1 = 6,
        K_2 = 6,
        tt = tau_T,
        tf = tau_F
    )
    print("状态: ", model.state)
    print("p: ", model.p)
    print("接入时延: ", model.access_delay)
    print("排队时延: ", model.queuing_delay)
    print("吞吐量: ", model.throughput)


