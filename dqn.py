#!/usr/bin/env python
import numpy as np
from optimizers import sgd
import theano
import theano.tensor as T
import json


__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

def _get_weights(name, shape1, shape2, init='nestrov'):
    if init == 'rand':
        x = np.random.randn(shape1, shape2) 
    elif init == 'nestrov':
        x = np.random.uniform(-np.sqrt(1. / shape2), np.sqrt(1. / shape2), (shape1, shape2))
    else:
        raise NotImplementedError("don't know how to initialize the weight matrix")
    return theano.shared(floatX(x), name)

def _get_zeros(name, shape1):
    x = 0.0 * np.random.rand(shape1,) 
    return theano.shared(floatX(x), name)

class DQN(object):
    def __init__(self, n_in, n_out, saved_weights = None):
        self.n_out = n_out
        self.n_in = n_in
        self._update = sgd 
        if saved_weights is None:
            w_i_o = _get_weights('W_i_h', self.n_in, self.n_out)
            w_target_i_o = _get_weights('W_target_i_h', self.n_in, self.n_out)
            #b_hidden = _get_zeros('B_h', n_hidden)
            #b_target_hidden = _get_zeros('B_target_h', n_hidden)
            #w_h_o = _get_weights('W_h_o', self.n_hidden, self.n_out)
            #w_target_h_o = _get_weights('W_h_o', self.n_hidden, self.n_out)
            b_out = _get_zeros('B_o', self.n_out)
            b_target_out = _get_zeros('B_target_o', self.n_out)
            self.params = [w_i_o, b_out]
            #self.reg_params = [w_i_h,  w_h_o]
            self.params_target = [w_target_i_o, b_target_out]
            #self.reg_params_target = [w_target_i_h,  w_target_h_o]
        else:
            _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
            assert len(_params) == 2
            w_i_o = theano.shared(_params[0], 'W_i_h')
            w_target_i_o = theano.shared(_params[0], 'W_target_i_h')
            #b_hidden = theano.shared(_params[1], 'B_h')
            #b_target_hidden = theano.shared(_params[1], 'B_target_h')
            #w_h_o = theano.shared(_params[2], 'W_h_o')
            #w_target_h_o = theano.shared(_params[2], 'W_target_h_o')
            b_out = theano.shared(_params[1], 'B_o')
            b_target_out = theano.shared(_params[1], 'B_target_o')
            self.params = [w_i_o, b_out]
            self.params_target = [w_target_i_o, b_target_out]
            #self.reg_params = [w_i_h,  w_h_o]
            #self.reg_params_target = [w_target_i_h,  w_target_h_o]
        self.__make_graph__()

    def save_weights(self, save_path):
        _params = json.dumps([i.tolist() for i in self.get_params()])
        f = open(save_path, 'w')
        f.write(_params)
        f.flush()
        f.close()
        return _params
    
    def __make_graph__(self,):
        gamma = T.fscalar('gamma') 
        lr = T.fscalar('lr') 
        S = T.fmatrix('S') #(batch_size, n_in)
        S_prime = T.fmatrix('S_prime') #(batch_size, n_in)
        A = T.ivector('A') #(batch_size)
        D = T.ivector('D') #(batch_size) (1 if is_terminal else 0)
        R = T.fvector('R') #(batch_size)
        U = T.fscalar('UR')

        def l2norm(t):
            return T.sqrt(T.sum(T.sqr(t)))
            
        w_i_o = self.params[0] #(n_in, n_hidden)
        b_out= self.params[1] #(n_hidden)
        #w_h_o = self.params[2] #(n_hidden, n_out)
        #b_out = self.params[3] #(n_out)
        #h = T.nnet.relu(S.dot(w_i_h) + b_hidden, alpha = self._leaky_relu) #(batch_size, n_hidden) #leaky relu
        #h = S.dot(w_i_h) + b_hidden #(batch_size, n_hidden) #linear
        Q = S.dot(w_i_o) + b_out #(batch_size, n_out)
        Q_a = Q[T.arange(0,A.shape[0]), A] #(batch_size)

        w_target_i_o = self.params_target[0]
        b_target_out = self.params_target[1]
        #w_target_h_o = self.params_target[2]
        #b_target_out = self.params_target[3]
        #h_prime = T.nnet.relu(S_prime.dot(w_target_i_h) + b_target_hidden, alpha = self._leaky_relu)
        #h_prime = S_prime.dot(w_target_i_h) + b_target_hidden #linear
        Q_prime = S_prime.dot(w_target_i_o) + b_target_out #(batch_size, n_out)
        max_a = T.argmax(Q_prime, axis=1) #select best actions from s_prime
        max_a_Q_prime = Q_prime[T.arange(0, max_a.shape[0]), max_a] #T.max(Q_prime, axis=1)
        target_Q = R + (gamma * (1 - D) * max_a_Q_prime)

        sqr_loss = T.mean(T.sqr(target_Q - Q_a))
        gw = T.grad(sqr_loss, w_i_o)
        gb = T.grad(sqr_loss, b_out)
        gg = T.sqrt(l2norm(gw) ** 2 + l2norm(gb) ** 2)
        #huber_loss = (T.sqrt(1 + T.sqr(target_Q - Q_a)) - 1).mean()

        self.get_params = theano.function(inputs = [], outputs = [T.as_tensor_variable(p) for p in self.params])
        self.get_grads = theano.function(inputs = [S, A, R, D, S_prime, gamma], outputs=[gw, gb, gg])
        self.get_Q = theano.function(inputs=[S], outputs=Q)
        self.get_Q_a = theano.function(inputs=[S, A], outputs=Q_a)
        self.get_Q_max_a = theano.function(inputs=[S], outputs=T.argmax(Q, axis=1))
        self.get_target_Q = theano.function(inputs=[S_prime, R, D, gamma], outputs=target_Q)
        self.get_sqr_loss = theano.function(inputs=[S, A, R, D, S_prime, gamma], outputs=sqr_loss)
        #self.get_huber_loss = theano.function(inputs=[S, A, R, D,  S_prime, gamma], outputs=huber_loss)
        self.do_sqr_loss_update = theano.function(inputs=[S, A, R, D, S_prime, gamma, lr], outputs = [sqr_loss, gg],
                                        updates = self._update(sqr_loss, self.params, lr))
        #self.do_huber_loss_update = theano.function(inputs=[S, A, R, D, S_prime, gamma], outputs = huber_loss, 
        #                                updates = self._update(huber_loss, self.params))
        self.update_target = theano.function(inputs=[U], 
                                        updates = [(param_t, (1 - U) * param_t + U * param) for param, param_t in zip(self.params_target, self.params)])
        self.copy_to_target = theano.function(inputs=[], updates = [(param_t, param) for param_t, param in zip(self.params_target, self.params)])
