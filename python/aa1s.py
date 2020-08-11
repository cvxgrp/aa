'''
Anderson Acceleration Class
'''
import numpy as np
import copy

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
            wrk['beta_0'] = 0.01 # mixing parameter of averaged iteration
            # storage of previous iterates
            wrk['aa'] = True # whether the previous AA candidate is accepted 
            wrk['x_aa'] = None # storage of the previous AA candidate (None if accepted)
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
            wrk['restart'] = [] # restart indices
            wrk['safeguard'] = [] # safeguard indices
        else:
            # WIP
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


    def safeguard(self, f, x, type='residual'):
        ''' safeguard checking
        Input:
            f: current iterate
            x: previous iterate (input to the update)
        Output:
            flag: accept AA candidate (True) or not (False)
        '''
        # residual decrease


        # objective decrease

        # domain checking

        # additional extensions can be added here ...

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
        if type1:
            # type1 AA update
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
            restart = wrk['restart'] 
            safeguard = wrk['safeguard']

            # return the fixed-point mapping result for the first iteration
            if iter_count == 0:
                x1 = fp(x)
                wrk['x_k_1'] = x
                wrk['g_k_1'] = x - x1
                return x1 

            m += 1
            if not aa:
                x_aa = wrk['x_aa']
                x1_aa = fp(x_aa)
                s0 = x_aa - 
            else:
                x1 = fp(x)


            s0 = 
            if iter_count == 0:
                x1 = fp(x)
            else:
                x1 = x - H_AA(g, H_vecs1, H_vecs2)

            ## check every M steps

            ## calculate successive differences
            s0 = x1 - x

            x_rec[:, i+1] = x1
            t_rec[i+1] = time.clock() - t0;
            s0 = x1 - x0
            fx0 = fx(x1)
            g1 = x1 - fx0
            y0 = g1 - g0
            x0 = x1
            g0 = g1

        else:
            # WIP
            pass



    '''
    Anderson Acceleration Iterations
    '''
    def aa1_iter(self, f, x):
        # param is a dictionary with indices 'itermax', 'mem_size', 
        # 'print_gap', 'theta', 'tau', 'safeguard_type', 'M' (check every M iterations)
        beta_0 = 0.01 # biased towards more progressive iterations
        itermax = param['itermax']
        print_gap = param['print_gap']
        n = len(x0)
        x_rec = np.zeros((n,itermax+1))
        x_rec[:, 0] = x0
        t_rec = np.zeros((itermax+1,))
        rec = dict()

        mem_size = param['mem_size']
        theta = param['theta']
        tau = param['tau']
        fx0 = fx(x0)
        g0 = x0 - fx0
        Shat_mem = []
        H_vecs1 = []
        H_vecs2 = []
        count = 0
        m = 0
        t0 = time.clock()
        t_rec[0] = 0
        rec['restart'] = []
        rec['safeguard'] = []

        for i in range(itermax):
            m += 1 # default increase of memory
            if i % print_gap == 0:
                print('###iteration = {}, residual norm = {}'.format(i, r_rec[i])
            if i == 0:
                x1 = fx0
            else:
                x1 = x0 - H_AA(g0, H_vecs1, H_vecs2)

            # update using the AA candidate
            s0 = 

            # check every M iteration

            ### Safeguard checking
            if safeguard():
                # this is the corrected safeguard according to the finished draft
                count += 1
            else:
                rec['safeguard'].append(i)
                print('In iteration {}, safeguard is used to produce the next iteration {}'.format(i, i+1))
                x1 = beta_0 * x0 + (1-beta_0) * x1_ipalm #(x0 - g0)
                u1 = x1[dims[0]:dims[1]]
                a1 = x1[dims[1]:dims[2]].reshape(n_nodes,n_nodes)
                b1 = x1[dims[2]:dims[3]][0]
                objective1_ll = objective(u1, a1, b1)
                objective1 = objective1_ll - reg * (np.linalg.norm(u1)**2 
                                                    + np.linalg.norm(a1)**2 + np.linalg.norm(b1)**2) 
            
            ### update iteration values
            x_rec[:, i+1] = x1
            t_rec[i+1] = time.clock() - t0;
            s0 = x1 - x0
            fx0 = fx(x1)
            g1 = x1 - fx0
            y0 = g1 - g0
            x0 = x1
            g0 = g1

            ### Restart checking
            if m <= mem_size:
                if len(Shat_mem) > 0: # only do when nonempty memory
                    s0hat = s0 - ((Shat_mem @ s0).transpose() @ Shat_mem).transpose()
                else:
                    s0hat = s0
                ### restart if not strongly independent
                if np.linalg.norm(s0hat) < tau * np.linalg.norm(s0):
                    rec['restart'].append(i)
                    s0hat = s0
                    m = 1
                    Shat_mem = []
                    H_vecs1 = []
                    H_vecs2 = []

            else: # memory exceeds
                s0hat = s0
                m = 1
                Shat_mem = []
                H_vecs1 = []
                H_vecs2 = []

            if len(Shat_mem) > 0:
                Shat_mem = np.vstack([Shat_mem, s0hat.transpose() / np.linalg.norm(s0hat)])
            else:
                Shat_mem = s0hat.transpose() / np.linalg.norm(s0hat)
                Shat_mem = Shat_mem.reshape(1, len(Shat_mem))

            ### Powell regularization
            gamma0 = s0hat.transpose() @ H_AA(y0, H_vecs1, H_vecs2) / (np.linalg.norm(s0hat)**2)
            theta0 = phi(gamma0, theta);
            y0tilde = theta0 * y0 - (1-theta0) * g0

            ### Update H_vecs
            Hytilde = H_AA(y0tilde, H_vecs1, H_vecs2)
            hvec1 = s0 - Hytilde
            hvec2 = H_AAt(s0hat, H_vecs1, H_vecs2) / (s0hat.transpose() @ Hytilde)
            if len(H_vecs1) > 0:
                H_vecs1 = np.hstack([H_vecs1, hvec1.reshape(len(hvec1),1)])
            else:
                H_vecs1 = np.hstack([H_vecs1, hvec1]).reshape(len(hvec1), 1)
            if len(H_vecs2) > 0:
                H_vecs2 = np.hstack([H_vecs2, hvec2.reshape(len(hvec2),1)])
            else:
                H_vecs2 = np.hstack([H_vecs2, hvec2]).reshape(len(hvec2), 1)
                
        return x_rec, r_rec, t_rec, rec