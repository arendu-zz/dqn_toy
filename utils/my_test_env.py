import numpy as np

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape
        #z = 1.0 / 255.0
        self.bad_state = np.uint8(np.random.randint(0, 5, shape))
        self.normal_state = np.uint8(np.random.randint(150, 155, shape))
        self.good_state = np.uint8(np.random.randint(245, 250, shape))
        self.states = [self.bad_state, self.normal_state, self.good_state]
        self.flatten_size = self.bad_state.flatten().shape[0]
        self.zero_state = np.uint8(np.zeros_like(self.bad_state))
        self.np_states = np.array(self.states)


class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    """
    def __init__(self, shape=(84, 84, 3), scale_state = False, flatten_history = False):
        #3 states
        self.history_len = 4
        self.rewards = [-0.1, 0, 0.1]
        self.cur_state = 0
        self.num_iters = 0
        self.state_history = []
        self.was_in_second = False
        self.action_space = ActionSpace(4)
        self.observation_space = ObservationSpace(shape)
        self.scale_state = scale_state
        self.flatten_history = flatten_history 

    def action_size(self,):
        return 4
    
    def get_observation(self,):
        return self.observation_space.states[self.cur_state]

    def get_weights(self,):
        padded_state_history = [-1] * (self.history_len - len(self.state_history))
        #print padded_state_history
        state_hist = self.state_history + padded_state_history
        #print state_hist
        _wt = np.array([self.observation_space.states[i] if i > -1 else self.observation_space.zero_state for i in state_hist])
        _wt = _wt.flatten()
        _wt = _wt.reshape(1, _wt.shape[0])
        if self.scale_state:
            _wt = np.float32(_wt) / 255.0 
        return _wt

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.state_history = []
        if self.flatten_history:
            return self.get_weights()
        else:
            return self.get_observation()
        

    def step(self, action):
        assert(0 <= action <= 3)
        self.num_iters += 1
        if action < 3:
            self.cur_state = action
        reward = self.rewards[self.cur_state]
        if self.was_in_second is True:
            reward *= -10
        if self.cur_state == 1:
            self.was_in_second = True
        else:
            self.was_in_second = False
        self.state_history.insert(0, self.cur_state)
        self.state_history = self.state_history[:self.history_len]
        if self.flatten_history:
            _wt = self.get_weights()
        else:
            _wt = self.get_observation() #self.get_weights()
        return _wt, float(reward), self.num_iters >= 5, {'ale.lives':0}


    def render(self):
        print(self.cur_state)
