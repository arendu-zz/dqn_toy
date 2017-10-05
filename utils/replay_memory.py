#!/usr/bin/env python
from theano import config as Tconfig
import numpy as np
__author__ = 'arenduchintala'

if Tconfig.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64

class ReplayMemory(object):
    def __init__(self,limit=1000):
        self.limit = limit
        self.SC= None
        self.A = None
        self.R = None
        self.T = None
        self.SN = None

    def get_size(self,):
        return 0 if self.A is None else self.A.shape[0]

    def add_experience(self, s_c, a, r, is_terminal, s_n):
        self.remove_experience()
        if self.SC is None:
            assert len(s_c.shape) == 2
            assert isinstance(a, int)
            assert isinstance(is_terminal, int)
            assert isinstance(r, float)
            assert s_c.shape[0] == 1
            self.SC = s_c
            self.SN = s_n
            self.A = np.array([a])
            self.R = np.array([r])
            self.T = np.array([is_terminal])
        else:
            self.SC = np.append(self.SC, s_c, axis = 0)
            self.SN = np.append(self.SN, s_n, axis = 0)
            self.R = np.append(self.R, r)
            self.A = np.append(self.A, a)
            self.T = np.append(self.T, is_terminal)
        assert self.SC.shape[0] == self.R.shape[0] == self.A.shape[0] == self.SN.shape[0] == self.T.shape[0]

    def remove_experience(self,):
        remove_num = self.get_size() - self.limit 
        if remove_num > 0:
            self.SC = np.delete(self.SC, range(remove_num), axis = 0)
            self.A = np.delete(self.A, range(remove_num), axis = 0)
            self.R = np.delete(self.R, range(remove_num), axis = 0)
            self.T = np.delete(self.T, range(remove_num), axis = 0)
            self.SN = np.delete(self.SN, range(remove_num), axis = 0)
            assert self.SC.shape[0] == self.R.shape[0] == self.A.shape[0] == self.SN.shape[0] == self.T.shape[0]
        else:
            pass

    def get_batch_experience(self, bs):
        batch_idx = np.random.choice(self.A.shape[0], bs, replace=False)
        return self.SC[batch_idx], self.A[batch_idx], self.R[batch_idx], self.T[batch_idx], self.SN[batch_idx]


