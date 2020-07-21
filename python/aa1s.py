'''
Anderson Acceleration Basic Functions
'''

def H_AA(x, Hv1, Hv2):
    ## supplementary function for AA-I(-safe) updates
    if len(Hv1) > 0 and len(Hv2) > 0:
        y = x + Hv1 @ (Hv2.transpose() @ x)
    else:
        y = x    
    return y
                       
def H_AAt(x, Hv1, Hv2):
    ## supplementary function for AA-I(-safe) updates
    if len(Hv1) > 0 and len(Hv2) > 0:
        y = x + Hv2 @ (Hv1.transpose() @ x)
    else:
        y = x
    return y

def phi(eta, theta):
    ## supplementary function for AA-I(-safe) updates (Powell regularization)
    if abs(eta) >= theta:
        thetay = 1
    elif eta != 0:
        thetay = (1-np.sign(eta)*theta) / (1-eta)
    else:
        thetay = 1 - theta
        
    return thetay


def safeguard():
    ## safeguard checking
    # residual decrease

    # objective decrease

    # domain checking

    # additional extensions can be added here ...

'''
Anderson Acceleration Iterations
'''
def aa1_iter(x0, param):
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