import numpy as np
import tensorflow as tf
from functools import partial
import pdb

def cost_matrix(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    "Returns the cosine distance"
    #NOTE: cosine distance and Euclidean distance
    # x_col = x.unsqueeze(1)
    # y_lin = y.unsqueeze(0)
    # c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    # return c
    x = tf.nn.l2_normalize(x, 1, epsilon=1e-12)
    y = tf.nn.l2_normalize(y, 1, epsilon=1e-12)
    tmp1 = tf.matmul(x, y, transpose_b=True)
    cos_dis = 1 - tmp1

    x_col = tf.expand_dims(x, 1)
    y_lin = tf.expand_dims(y, 0)
    res = tf.reduce_sum(tf.abs(x_col - y_lin), 2)

    return cos_dis


def IPOT(C, n, m, beta=0.5):

    # sigma = tf.scalar_mul(1 / n, tf.ones([n, 1]))

    sigma = tf.ones([m, 1]) / tf.cast(m, tf.float32)
    T = tf.ones([n, m])
    A = tf.exp(-C / beta)
    for t in range(50):
        Q = tf.multiply(A, T)
        for k in range(1):
            delta = 1 / (tf.cast(n, tf.float32) * tf.matmul(Q, sigma))
            sigma = 1 / (
                tf.cast(m, tf.float32) * tf.matmul(Q, delta, transpose_a=True))
        # pdb.set_trace()
        tmp = tf.matmul(tf.diag(tf.squeeze(delta)), Q)
        T = tf.matmul(tmp, tf.diag(tf.squeeze(sigma)))
    return T


def IPOT_np(C, beta=0.5):
        
    n, m = C.shape[0], C.shape[1]
    sigma = np.ones([m, 1]) / m
    T = np.ones([n, m])
    A = np.exp(-C / beta)
    for t in range(20):
        Q = np.multiply(A, T)
        for k in range(1):
            delta = 1 / (n * (Q @ sigma))
            sigma = 1 / (m * (Q.T @ delta))
        # pdb.set_trace()
        tmp = np.diag(np.squeeze(delta)) @ Q
        T = tmp @ np.diag(np.squeeze(sigma))
    return T

def IPOT_distance(C, n, m):
    T = IPOT(C, n, m)
    distance = tf.trace(tf.matmul(C, T, transpose_a=True))
    return distance


def shape_list(x):
   """Return list of dims, statically where possible."""
   x = tf.convert_to_tensor(x)
   # If unknown rank, return dynamic shape
   if x.get_shape().dims is None:
       return tf.shape(x)
   static = x.get_shape().as_list()
   shape = tf.shape(x)
   ret = []
   for i in range(len(static)):
       dim = static[i]
       if dim is None:
           dim = shape[i]
       ret.append(dim)
   return ret


def IPOT_alg(C, beta=1, t_steps=10, k_steps=1):
   b, n, m = shape_list(C)
   sigma = tf.ones([b, m, 1]) / tf.cast(m, tf.float32)  # [b, m, 1]
   T = tf.ones([b, n, m])
   A = tf.exp(-C / beta)  # [b, n, m]
   for t in range(t_steps):
       Q = A * T  # [b, n, m]
       for k in range(k_steps):
           delta = 1 / (tf.cast(n, tf.float32) *
                        tf.matmul(Q, sigma))  # [b, n, 1]
           sigma = 1 / (tf.cast(m, tf.float32) * tf.matmul(Q,
                                                           delta, transpose_a=True))  # [b, m, 1]
       T = delta * Q * tf.transpose(sigma, [0, 2, 1])  # [b, n, m]
#    distance = tf.trace(tf.matmul(C, T, transpose_a=True))
   return T

def IPOT_distance2(C, beta=1, t_steps=10, k_steps=1):
   b, n, m = shape_list(C)
   sigma = tf.ones([b, m, 1]) / tf.cast(m, tf.float32)  # [b, m, 1]
   T = tf.ones([b, n, m])
   A = tf.exp(-C / beta)  # [b, n, m]
   for t in range(t_steps):
       Q = A * T  # [b, n, m]
       for k in range(k_steps):
           delta = 1 / (tf.cast(n, tf.float32) * tf.matmul(Q, sigma))  # [b, n, 1]
           sigma = 1 / (tf.cast(m, tf.float32) * tf.matmul(Q, delta, transpose_a=True))  # [b, m, 1]
       T = delta * Q * tf.transpose(sigma, [0, 2, 1])  # [b, n, m]
   distance = tf.trace(tf.matmul(C, T, transpose_a=True))
   return distance


def GW_alg(Cs, Ct, beta=0.5, iteration=5, OT_iteration=20):
    bs, _, n = shape_list(Cs)
    _, _, m = shape_list(Ct)
    one_m = tf.ones([bs, m, 1]) / tf.cast(m, tf.float32)
    one_n = tf.ones([bs, n, 1]) / tf.cast(n, tf.float32)
    p = tf.ones([bs, m, 1]) / tf.cast(m, tf.float32)
    q = tf.ones([bs, n, 1]) / tf.cast(n, tf.float32)
    
    Cst = tf.matmul(tf.matmul(Cs**2, q), one_m, transpose_b=True) +  \
        tf.matmul(one_n, tf.matmul(p, Ct**2, transpose_a=True, transpose_b=True))

    gamma = tf.matmul(q, p, transpose_b=True)

    for i in range(iteration):
        
        tmp1 = tf.matmul(Cs, gamma)
        C_gamma = Cst - 2 * tf.matmul(tmp1, Ct, transpose_b=True)

        gamma = IPOT_alg(C_gamma, beta=beta, t_steps=OT_iteration)
    Cgamma = Cst - 2 * tf.matmul(
        tf.matmul(Cs, gamma), tf.transpose(Ct, [0, 2, 1]))
    # pdb.set_trace()
    return gamma, Cgamma

def prune(dist, beta=0.1):
    
    min_score = tf.reduce_min(dist, axis=[1, 2], keepdims=True)
    max_score = tf.reduce_max(dist, axis=[1, 2], keepdims=True)
    # pdb.set_trace()
    # min_score = dist.min()
    # max_score = dist.max()
    threshold = min_score + beta * (max_score - min_score)
    res = dist - threshold
    return tf.nn.relu(res)

# def GW_alg_abs(Cs, Ct, beta=0.5, iteration=5, OT_iteration=20):
#     bs, _, n = shape_list(Cs)
#     _, _, m = shape_list(Ct)
#     one_m = tf.ones([bs, m, 1]) / tf.cast(m, tf.float32)
#     one_n = tf.ones([bs, n, 1]) / tf.cast(n, tf.float32)
#     p = tf.ones([bs, m, 1]) / tf.cast(m, tf.float32)
#     q = tf.ones([bs, n, 1]) / tf.cast(n, tf.float32)

#     # Cst = tf.matmul(tf.matmul(Cs**2, q), one_m, transpose_b=True) +  \
#     #     tf.matmul(one_n, tf.matmul(
#     #         p, Ct**2, transpose_a=True, transpose_b=True))
#     Cst = tf.matmul(tf.matmul(Cs**2, q), one_m, transpose_b=True) +  \
#         tf.matmul(one_n, tf.matmul(
#             p, Ct**2, transpose_a=True, transpose_b=True))

#     gamma = tf.matmul(q, p, transpose_b=True)

#     for i in range(iteration):

#         tmp1 = tf.matmul(Cs, gamma)
#         C_gamma = Cst - 2 * tf.matmul(tmp1, Ct, transpose_b=True)

#         gamma = IPOT_alg(C_gamma, beta=beta, t_steps=OT_iteration)
#     Cgamma = Cst - 2 * tf.matmul(
#         tf.matmul(Cs, gamma), tf.transpose(Ct, [0, 2, 1]))
#     # pdb.set_trace()
#     return gamma, Cgamma

def GW_distance(Cs, Ct, beta=0.5, iteration=5, OT_iteration=20):
    T, Cst = GW_alg(Cs, Ct, beta=beta, iteration=iteration, OT_iteration=OT_iteration)
    GW_distance = tf.trace(tf.matmul(Cst, T, transpose_a=True))
    return GW_distance


def FGW_distance(Cs, Ct, C, beta=0.5, iteration=5, OT_iteration=20):
    T, Cst = GW_alg(Cs, Ct, beta=beta, iteration=iteration,
                    OT_iteration=OT_iteration)
    GW_distance = tf.trace(tf.matmul(Cst, T, transpose_a=True))
    W_distance = tf.trace(tf.matmul(C, T, transpose_a=True))
    return GW_distance, W_distance

# def cos_batch_torch(x, y):
# 	"Returns the cosine distance batchwise"
# 	# x is the image feature: bs * d * m * m
# 	# y is the audio feature: bs * d * nF
# 	# return: bs * n * m
# 	# print(x.size())
# 	bs = x.size(0)
# 	D = x.size(1)
# 	assert(x.size(1) == y.size(1))
# 	x = x.contiguous().view(bs, D, -1)  # bs * d * m^2
# 	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
# 	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
# 	cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
# 	cos_dis = 1 - cos_dis  # to minimize this value
# 	# return cos_dis.transpose(2,1)
# 	beta = 0.1
# 	min_score = cos_dis.min()
# 	max_score = cos_dis.max()
# 	threshold = min_score + beta * (max_score - min_score)
# 	res = cos_dis - threshold
# 	# torch.nn.functional.relu(res.transpose(2,1))
# 	return torch.nn.functional.relu(res.transpose(2, 1))

# def GW_distance(Cs, Ct, p, q, lamda=0.5, iteration=5, OT_iteration=20):
# 	'''
# 	:param X, Y: Source and target embeddings , batchsize by embed_dim by n
# 	:param p, q: probability vectors
# 	:param lamda: regularization
# 	:return: GW distance
# 	'''
# 	Cs = cos_batch_torch(X, X).float().cuda()
# 	Ct = cos_batch_torch(Y, Y).float().cuda()
# 	# pdb.set_trace()
# 	bs = Cs.size(0)
# 	m = Ct.size(2)
# 	n = Cs.size(2)
# 	T, Cst = GW_alg(Cs, Ct, bs, n, m, p, q, beta=lamda,
#                  iteration=iteration, OT_iteration=OT_iteration)
# 	temp = torch.bmm(torch.transpose(Cst, 1, 2), T)
# 	distance = batch_trace(temp, m, bs)
# 	return distance
