import numpy as np
import tensorflow as tf
def TVR_Diff_tf(y,alph = 1,dx = 1):
    n=len(y)
    d0 = -np.ones(n)/dx
    du = np.ones(n-1)/dx
    dl = np.zeros(n-1)
    dl[-1] = d0[-1]
    d0[-1] *= -1
    D = tf.math.add_n([tf.linalg.diag(tf.constant(dl),k=-1),tf.linalg.diag(tf.constant(d0),k=0),tf.linalg.diag(tf.constant(du),k=1)])
    DT = tf.transpose(D)
    def A(x): return (tf.cumsum(x) - 0.5 * (x + x[0])) * dx
    def A(x): return (tf.cumsum(x) - 0.5 * (x + x[0])) * dx
    def AT(x): 
        a  =tf.math.reduce_sum(x[1:])/2.0
        b = (tf.math.reduce_sum(x)-tf.cumsum(x)+0.5*x)[1:]
        return tf.concat([tf.reshape(a,[1,1]),b],axis = 0)
    cgtol= 1e-4
    diffkernel = 'abs'
    ep = 10**-4
    cgmaxit = 100
    u0 = tf.matmul(D,tf.reshape(y,[n,1]),a_is_sparse=True)
    u = tf.identity(u0)
    ofst = y[0]
    ATb = AT(ofst - tf.reshape(y,[n,1]))
    for ii in range(1, 100+1):
        if diffkernel == 'abs':
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = tf.linalg.diag(tf.reshape(1/tf.sqrt((tf.matmul(D,u,a_is_sparse=False))**2 + ep),[n]),k=0)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = dx *tf.matmul(tf.matmul(DT, Q) , D)
        elif diffkernel == 'sq':
            L = dx * DT * D
        else:
            raise ValueError('Invalid diffkernel value')

        # Gradient of functional.
        g = AT(A(u)) + ATb + alph * tf.matmul(L, u)

        # Prepare to solve linear equation.
        precondflag = True
        if precondflag:
            # Simple preconditioner.
            P = alph * tf.linalg.diag(tf.linalg.diag_part(L) + 1, 0, n, n)
        else:
            P = None

        B =tf.linalg.diag(tf.reshape(alph * tf.matmul(L, u) + AT(A(u)),[n]))
        s = tf.linalg.solve(B,g)

        # Updating the solution
        u = u - s
    return u