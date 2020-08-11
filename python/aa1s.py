'''
Anderson Acceleration Class
'''
import numpy as np
from numpy.linalg import norm

class AndersonAccelerator:

    def __init__(self, dim, mem, params, type1=True, safeguard_type='residual'):
        wrk = dict()
        ## load basic (hyper-)parameters
        wrk['dim'] = dim
        wrk['mem'] = mem
        wrk['type1'] = type1
        wrk['safeguard_type'] = safeguard_type
        if type1:
            ## load stabilization (hyper-)parameters
            wrk['theta'] = params['theta']
            wrk['tau'] = params['tau']
            wrk['D'] = params['D']
            wrk['eps'] = params['eps']
            wrk['beta_0'] = params['beta_0'] # mixing parameter of averaged iteration
            wrk['Ubar'] = None # store norm of g0
            # storage of previous iterates
            wrk['aa'] = True # whether the previous AA candidate is accepted 
            wrk['x_aa'] = None # storage of the previous AA candidate (None if accepted, i.e., aa = True)
            wrk['x_k_1'] = None # storage of x_{k-1} (None for the first iteration)
            wrk['g_k_1'] = None # storage of g_{k-1} (None for the first iteration)
            ## index counts
            wrk['iter_cnt'] = 0 # iteration count
            wrk['m'] = 0 # current memory
            wrk['aa_cnt'] = 0 # current AA update count
            ## create intermediate quantity placeholders
            wrk['Shat_mem'] = [] # Shat matrix (each column is \hat{s}_k)
            wrk['H_vecs1'] = [] # intermediate quantity for matrix-free AA updates
            wrk['H_vecs2'] = [] # intermediate quantity for matrix-free AA update
            ## result storage
            wrk['rec_restart'] = [] # restart indices
            wrk['rec_safeguard'] = [] # safeguard indices
        else:
            # WIP: non-convex momentum type1 and type2 
            pass
        self._wrk = wrk


    ## AA helper functions
    def H_AA(self, x, Hv1, Hv2):
        ## supplementary function for AA-I(-safe) updates
        if len(Hv1) > 0 and len(Hv2) > 0:
            y = x + Hv1 @ (Hv2.transpose() @ x)
        else:
            y = x    
        return y
                           
    def H_AAt(self, x, Hv1, Hv2):
        ## supplementary function for AA-I(-safe) updates
        if len(Hv1) > 0 and len(Hv2) > 0:
            y = x + Hv2 @ (Hv1.transpose() @ x)
        else:
            y = x
        return y

    def phi(self, eta, theta):
        ## supplementary function for AA-I(-safe) updates (Powell regularization)
        if abs(eta) >= theta:
            thetay = 1
        elif eta != 0:
            thetay = (1-np.sign(eta)*theta) / (1-eta)
        else:
            thetay = 1 - theta
            
        return thetay


    def safeguard(self, wrk, type='residual'):
        ''' safeguard checking
        Input:
            wrk: workspace that stores information needed for safeguard checking
        Output:
            flag: accept AA candidate (True) or not (False)
        '''
        if type == 'residual':
            # check residual decrease
            D = wrk['D']
            Ubar = wrk['Ubar']
            eps = wrk['eps']
            g = wrk['g_k_1']
            aa_cnt = wrk['aa_cnt']
            if norm(g) <= D * Ubar * (aa_cnt + 1)**(-1-eps):
                return True
            else:
                return False

        else:
            ## additional extensions can be added here ...
            # objective decrease
            # domain checking
            # etc.
            pass

    ## AA main update
    def apply(self, fp, x):
        ''' AA main update
        Input: 
            fp: fixed point mapping
            x: input iterate to the update

        Output:
            x1: next iterate
        '''
        ## load the data and (hyper-)parameters
        wrk = self._wrk 
        ## check type1 or type2
        type1 = wrk['type1']
        safeguard_type = wrk['safeguard_type']
        if type1:
            ## type1 AA update
            dim = wrk['dim']
            mem = wrk['mem']
            theta = wrk['theta']
            tau = wrk['tau']
            beta_0 = wrk['beta_0']
            aa = wrk['aa']
            x_aa = wrk['x_aa']
            x_k_1 = wrk['x_k_1']
            g_k_1 = wrk['g_k_1']
            iter_cnt = wrk['iter_cnt']
            m = wrk['m']
            aa_cnt = wrk['aa_cnt']
            Shat_mem = wrk['Shat_mem']
            H_vecs1 = wrk['H_vecs1'] 
            H_vecs2 = wrk['H_vecs2']
            rec_restart = wrk['rec_restart'] 
            rec_safeguard = wrk['rec_safeguard']

            ## return the fixed-point mapping result for the first iteration
            if iter_cnt == 0:
                x1 = beta_0*x + (1-beta_0)*fp(x) # KM iteration
                g = x - x1
                wrk['x_k_1'] = x
                wrk['g_k_1'] = g
                wrk['Ubar'] = norm(g)
                wrk['iter_cnt'] = iter_cnt + 1
                #self._wrk = wrk
                #print(x)
                return x1 

            #print(iter_cnt, x_k_1)

            m += 1
            x1 = beta_0*x + (1-beta_0)*fp(x) # KM iteration
            g = x - x1
            wrk['x_k_1'] = x
            wrk['g_k_1'] = g

            if not aa:
                # previous AA candidate not accepted
                x1_aa = fp(x_aa)
                g_aa = x_aa - x1_aa
                s_k_1 = x_aa - x_k_1
                y_k_1 = g_aa - g_k_1
            else:
                # previous AA candidate accepted
                s_k_1 = x - x_k_1
                y_k_1 = g - g_k_1
                # print(x_k_1)

            ## Restart checking
            if m <= mem:
                if len(Shat_mem) > 0: # only do when nonempty memory
                    s_k_1_hat = s_k_1 - ((Shat_mem @ s_k_1).transpose() @ Shat_mem).transpose()
                else:
                    s_k_1_hat = s_k_1
                ### restart if not strongly independent
                if np.linalg.norm(s_k_1_hat) < tau * np.linalg.norm(s_k_1):
                    #print('restarted!!!')
                    rec_restart.append(iter_cnt)
                    wrk['rec_restart'] = rec_restart
                    s_k_1_hat = s_k_1
                    m = 1
                    Shat_mem = []
                    H_vecs1 = []
                    H_vecs2 = []

            else: 
                # memory exceeds
                #print('memory exceeds')
                s_k_1_hat = s_k_1
                m = 1
                Shat_mem = []
                H_vecs1 = []
                H_vecs2 = []

            if len(Shat_mem) > 0:
                Shat_mem = np.vstack([Shat_mem, s_k_1_hat.transpose() / np.linalg.norm(s_k_1_hat)])
            else:
                Shat_mem = s_k_1_hat.transpose() / np.linalg.norm(s_k_1_hat)
                Shat_mem = Shat_mem.reshape(1, len(Shat_mem))

            ## Powell regularization
            gamma_k_1 = s_k_1_hat.transpose() @ self.H_AA(y_k_1, H_vecs1, H_vecs2) / (np.linalg.norm(s_k_1_hat)**2)
            theta_k_1 = self.phi(gamma_k_1, theta);
            y_k_1_tilde = theta_k_1 * y_k_1 - (1-theta_k_1) * g_k_1
            #print('y_k_1_tilde - y_k_1 = {}'.format(norm(y_k_1_tilde-y_k_1)))

            ## Update H_vecs
            Hytilde = self.H_AA(y_k_1_tilde, H_vecs1, H_vecs2)
            hvec1 = s_k_1 - Hytilde
            hvec2 = self.H_AAt(s_k_1_hat, H_vecs1, H_vecs2) / (s_k_1_hat.transpose() @ Hytilde)
            if len(H_vecs1) > 0:
                H_vecs1 = np.hstack([H_vecs1, hvec1.reshape(len(hvec1),1)])
            else:
                H_vecs1 = np.hstack([H_vecs1, hvec1]).reshape(len(hvec1), 1)
            if len(H_vecs2) > 0:
                H_vecs2 = np.hstack([H_vecs2, hvec2.reshape(len(hvec2),1)])
            else:
                H_vecs2 = np.hstack([H_vecs2, hvec2]).reshape(len(hvec2), 1)

            ## AA candidate update
            x_aa = x - self.H_AA(g, H_vecs1, H_vecs2)

            # ## debug
            # x_aa_tmp = x - g - (s_k_1 - y_k_1)*np.dot(s_k_1, g) / np.dot(s_k_1,y_k_1)
            # print('x_aa - x_aa_tmp = {}'.format(norm(x_aa - x_aa_tmp)))
            # print('norm of x_aa_tmp = {}'.format(norm(x_aa_tmp)))

            ## Update workspace
            wrk['iter_cnt'] = iter_cnt + 1
            wrk['m'] = m
            wrk['Shat_mem'] = Shat_mem
            wrk['H_vecs1'] = H_vecs1
            wrk['H_vecs2'] = H_vecs2

            ## safeguard checking 
            # generalize to check every M steps like in a2dr??? 
            if not self.safeguard(wrk, safeguard_type):
                # AA candidate rejected
                #print('iter = {}; AA not pass'.format(iter_cnt))
                #print('safeguarded!!!')
                rec_safeguard.append(iter_cnt)
                wrk['rec_safeguard'] = rec_safeguard
                wrk['aa'] = False
                wrk['x_aa'] = x_aa
                #self._wrk = wrk
                return x1
            else:
                # AA candidate to be accepted
                #print('iter = {}; AA passed!'.format(iter_cnt))
                wrk['aa'] = True
                wrk['aa_cnt'] = aa_cnt + 1
                #self._wrk = wrk
                return x_aa

        else:
            # WIP: non-convex momentum type1 and type2 
            pass
