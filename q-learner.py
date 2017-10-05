#!/usr/bin/env python
import argparse
import pdb
import numpy as np
from theano import config as Tconfig
#from replay_memory import ReplayMemory
#from my_test_env import EnvTest
from utils.replay_buffer import ReplayBuffer
from utils.test_env import EnvTest
from dqn import DQN
np.set_printoptions(precision = 3, suppress=True)
__author__ = 'arenduchintala'

if Tconfig.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64

OBS_SIZE = (10, 10, 1)

def save_fig(e_list, rpe_list, eval_rpe_list, rl_list):
    if options.save_fig != '':
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(e_list, rpe_list, 'b', alpha= 0.5)
        #ax1.plot(e_list, eps_t_list, 'r-', alpha= 0.5)
        ax1.plot(e_list, eval_rpe_list, 'g')
        ax1.set_ylabel('Rewards per Episode')
        ax1.set_xlabel('epochs')
        ax2 = ax1.twinx()
        ax2.plot(e_list, rl_list, 'r', alpha=0.5)
        ax2.set_ylabel('loss (moving ave)', color='r')
        plt.savefig(options.save_fig, dpi=200)


def decay(start_val, end_val, t, final_t):
    if t == 0:
        return start_val
    elif 0 < t < final_t:
        return ((start_val - end_val) / (0.0 - final_t)) * t + start_val
    else:
        return end_val

def evaluate(_env, _dqn, reps=1):
    eval_r = []
    for _ in xrange(reps):
        #_env = EnvTest(shape=OBS_SIZE, scale_state=True, flatten_history = True)
        s = _env.reset()
        is_t = False
        summary = []
        rep_r = 0.
        while not is_t:
            idx = replay_memory.store_frame(s)
            q_state = replay_memory.encode_recent_observation()
            flat_q_state = floatX(q_state.flatten()[np.newaxis,:]) / 255.0
            _a = _dqn.get_Q_max_a(flat_q_state)[0]
            next_s, r, is_t, _ = _env.step(_a)
            rep_r += r
            replay_memory.store_effect(idx, _a, r, is_t)
            summary.append(str(a) + '[' + '%.4f'%r + ']')
            s = next_s
        eval_r.append(rep_r)
    return np.mean(eval_r), ' -> '.join(summary)


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="Q-learning")
    opt.add_argument('--start_eps', action='store', dest='start_eps', default=1.0, type=float)
    opt.add_argument('--end_eps', action='store', dest='end_eps', default=0.01, type=float)
    opt.add_argument('--start_lr', action='store', dest='start_lr', default=0.05, type=float)
    opt.add_argument('--end_lr', action='store', dest='end_lr', default=0.01, type=float)
    opt.add_argument('--num_steps', action='store', dest='NUM_STEPS', default=10000, type=int)
    opt.add_argument('--episode_length', action='store', dest='EPISODE_LENGTH', default=5, type=int)
    opt.add_argument('--batch_size', action='store', dest='BATCH_SIZE', default=32, type=int)
    opt.add_argument('--history_size', action='store', dest='HISTORY_SIZE', default=4, type=int)
    opt.add_argument('--fig', action='store', dest='save_fig', default = 'tmp.png')
    options = opt.parse_args()
    print options
    #env = EnvTest(shape=OBS_SIZE, scale_state=True, flatten_history = True)
    env = EnvTest(OBS_SIZE) 
    #replay_memory = ReplayMemory(limit = 1000)
    replay_memory = ReplayBuffer(1000, options.HISTORY_SIZE) 
    flat_inp_size = options.HISTORY_SIZE * OBS_SIZE[0] * OBS_SIZE[1] * OBS_SIZE[2]
    dqn = DQN(n_in = flat_inp_size, n_out = env.action_space.n)
    gamma = 0.99
    eps_t = options.start_eps
    lr_t = options.start_lr

    rpe_list = []
    e_list = []
    eval_rpe_list = []
    eps_t_list = []
    rl_list = []
    #gold = [1, 0, 1, 0, 2]
    _t = 0
    _e = 0
    running_loss_ave = []
    while _t < options.NUM_STEPS:
        rpe = 0.0
        _e += 1
        a_list = []
        state = env.reset()
        is_terminal = False
        while not is_terminal:
            _t += 1# = (e_idx * options.EPISODE_LENGTH) + t_idx
            idx = replay_memory.store_frame(state)
            q_state = replay_memory.encode_recent_observation()
            flat_q_state = floatX(q_state.flatten()[np.newaxis,:]) / 255.0
            q = dqn.get_Q(flat_q_state) #state should be (batch_size, n_in) and q should be (batch_size, n_out)
            if np.random.rand() < eps_t:
                a = np.random.choice(env.action_space.n, 1, replace=False)[0] # random action
                a_type='r'
            else:
                a = q.argmax() #best action
                a_type='g'
            #a = gold[(_t - 1) % 5]
            #a_type=' '
            next_state, r, is_terminal, _  = env.step(a)
            assert (_t % 5 == 0) == is_terminal
            a_list.append(str(a)+'(' +str(r) +',' +a_type +')')
            rpe += r
            #replay_memory.add_experience(state, a, r, 1 if is_terminal else 0, next_state)
            replay_memory.store_effect(idx, a, r, is_terminal)
            state = next_state
            if  _t % 4 == 0 and _t > 200:
                s_b, a_b, r_b, s_n_b, t_b = replay_memory.sample(options.BATCH_SIZE)
                flat_s_b = floatX(s_b.reshape((s_b.shape[0], -1))) / 255.0 #converts (batch_size, x_size,y_size,c_size) to (batch_size, |x * y * c * t|)
                flat_s_n_b = floatX(s_n_b.reshape(s_n_b.shape[0], -1)) / 255.0
                a_b = intX(a_b)
                r_b = floatX(r_b)
                t_b = intX(t_b)
                #_q_vals = dqn.get_Q_a(s_b, a_b)
                #_q_target_vals = dqn.get_target_Q(s_n_b, r_b, t_b, gamma)
                loss_b, grad_norm = dqn.do_sqr_loss_update(flat_s_b, a_b, r_b, t_b, flat_s_n_b, floatX(gamma), floatX(lr_t))
                running_loss_ave.insert(0,loss_b) 
                running_loss_ave = running_loss_ave[:100]
                #eval_r, eval_seq = evaluate(env, dqn)
                print  _t, 'loss:', '%.4f'%loss_b, 'grad_norm', '%.4f'%grad_norm, 'lr_t', '%.4f'%lr_t, 'eps_t', '%.4f'%eps_t
                if np.isnan(loss_b):
                    raise BaseException("Loss is nan!")
                if _t > 200 and _t % 50 == 0:
                    eps_t = decay(options.start_eps, options.end_eps, _t,  options.NUM_STEPS * 0.5)
                    lr_t = decay(options.start_lr, options.end_lr, _t, options.NUM_STEPS * 0.5)
            if _t % 500 == 0:
                dqn.copy_to_target()
        if _e % 5 == 0 and _t > 200:
            eval_r, eval_seq = evaluate(env, dqn, reps = 1)
            rl = np.mean(running_loss_ave)
            print 'Eval', '%.4f'%eval_r, 'Loss moving ave:', '%.4f'%rl
            rpe_list.append(rpe)
            eval_rpe_list.append(eval_r)
            eps_t_list.append(eps_t)
            rl_list.append(rl)
            e_list.append(_e)
        else:
            pass

    save_fig(e_list, rpe_list, eval_rpe_list, rl_list)
