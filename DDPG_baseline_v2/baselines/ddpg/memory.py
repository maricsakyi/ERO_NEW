import pickle

import numpy as np
import random
import math
import sortedcontainers

# Managing the replay buffer
class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

        self._next_idx = 0
        #self.alpha=0.6
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        #if self.length < self.maxlen:
        # We have space, simply increase the length.
        self.length += 1
        #print("max value:", np.max(self.data))
        #print("max position",np.argmax(self.data))

        self.data[self._next_idx] = v
        self._next_idx = (self.start + self.length) % self.maxlen
                 
        
        #print(np.argwhere(self.data==(np.random.choice(self.data,1,p=prob1))))
            
        '''elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
            #print(self.data[2])
            #creating probabilities our of rewards and using alpha to control the uniformity of the probability
            prob1=(self.data-np.min(self.data))/(np.max(self.data)-np.min(self.data))
            prob1=prob1**self.alpha
            prob1=prob1/sum(prob1)
            k=np.argwhere(self.data==(np.random.choice(self.data,1,p=prob1))).reshape(1)
            print(k)
            self.data[self._next_idx] = v
            self._next_idx = (self.start + self.length) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        #self.data[(self.start + self.length - 1) % self.maxlen] = v
       
        self.data[self._next_idx] = v
        self._next_idx = (self.start + self.length) % self.maxlen
        #print("V=",v)
        #print("_________________________________________________")
        #print("Index [0]:",self.data[0])
        #print("_________________________________________________")'''


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)



class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)


    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

        
    @property
    def nb_entries(self):
        return len(self.observations0)

class DynamicMemory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit
        self.sampled_inxs = np.array([])

        self.max_sigmoid = 6
        self.table_size = 10000000
        self.count=0
        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)
        #self.epsilon=0.88
        self.alpha=0.6

        # Used to record current accumulative reward, used to scale features
        self.current_accu_reward = -9999

        # Initilize the probilities which are used for sampling
        #self.probs = [0. for _ in range(self.limit)]
        self.probs = np.full(self.limit, 0.)

        # Store features
        self.td_errors = np.zeros(self.limit)
        self.transition_rewards = np.zeros(self.limit)
        self.accu_rewards = np.zeros(self.limit)
        #self.td_errors = [0. for _ in range(self.limit)]
        #self.transition_rewards = [0. for _ in range(self.limit)]
        #self.accu_rewards = [0. for _ in range(self.limit)]
        self.sigmoid = self.init_fastsigmoid()

    # Sample the indexes used for training
    # I is like a mask to mask out the transitions
    def sample_inxs(self, batch_size):
        if self.nb_entries <= batch_size:
            #I = [1 for i in range(self.nb_entries)]
            I = np.full(self.nb_entries, 1)
        else:
            #I = [1 if random.random() < self.probs[i] else 0 for i in range(self.nb_entries)]
            normal_dist = np.random.rand(self.nb_entries)
            I = np.greater(self.probs[:self.nb_entries], normal_dist)

        #self.sampled_inxs = [x for x in range(self.nb_entries) if I[x] == 1]
        self.sampled_inxs = np.argwhere(I==1)

        diff = batch_size - self.sampled_inxs.shape[0]
        if diff > 0:
            c_inxs = [x for x in range(self.nb_entries) if x not in self.sampled_inxs]
            if len(c_inxs) > diff:
                add_inxs = np.random.choice(c_inxs, diff)
                #add_inxs = np.array(random.sample(c_inxs, diff))
                #self.sampled_inxs = np.hstack(self.sampled_inxs,add_inxs)
                self.sampled_inxs = np.hstack((self.sampled_inxs.flatten(), add_inxs))
                self.sampled_inxs = np.reshape(self.sampled_inxs, (self.sampled_inxs.shape[0],1))
        return I

    # Randomly sample a batch from the sampled indexes
    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        #batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)
        if len(self.sampled_inxs) == 0:
        #if True:
            batch_idxs = np.random.choice(self.nb_entries, batch_size)
            #batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        else:
            batch_idxs = np.random.choice(self.sampled_inxs.flatten(), batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'batch_idxs': array_min2d(batch_idxs),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        idx = self.next_idx

        if self.observations0.length < self.observations0.maxlen:
            self.observations0.append(obs0)
            self.actions.append(action)
            self.rewards.append(reward)
            
            #k=self.rewards.data.flatten()
            #f=np.random.choice(k,1)
            #k=np.argwhere(k==f)
            #k=k.flatten()
            #print(k.item(0))
            #print(type(self.rewards.data))
            #print("reward:",min(self.rewards.data),"position:",np.argmin(self.rewards.data))
            self.observations1.append(obs1)
            self.terminals1.append(terminal1)
                #print(self.rewards.length," ",self.rewards.maxlen) 

        elif self.rewards.length == self.rewards.maxlen:
            print("function was called")
            
            idxx=self.new_append()
            self.new_append
            self.observations0.data[idxx]=obs0
            self.observations0.data[idxx]=obs0

            self.observations1.data[idxx]=obs1
            self.terminals1.data[idxx]=terminal1
            self.rewards.data[idxx]=reward
        else:
            # This should never happen.
            raise RuntimeError()
            
        # Storing probs
        self.probs[idx] = 1

        # Storing features
        self.td_errors[idx] = 0.5 # Initilize to 0.5, will be upadted later
        #self.transition_rewards[idx] = self.fastsigmoid(np.array([reward]))[0]
        self.transition_rewards[idx] = sigmoid(reward)
        
        #print("sigmoid reward:", sigmoid(reward), "fast sigmoid reward:",self.fastsigmoid(np.array([reward])[0]))
        self.accu_rewards[idx] = 0 # Initilize to 0, will be set later
        
    def new_append(self):
            # No space, "remove" the first item.
            self.count+=1
            print("inside:[",self.count,"]")
            keep=self.rewards.data.flatten()
            '''if self.count%2000==0:
                k=np.argmin(keep)
            else:
                k=np.argmax(keep)'''
            
           

            #creating probabilities of rewards and using alpha to control the uniformity of the probability
            prob1=(keep- min(keep))/(max(keep)-np.min(keep))
            prob1=np.exp(-2.0*0.5*prob1)
            prob1=prob1/sum(prob1)
            #print(prob1)
            k=np.argwhere(keep==(np.random.choice(keep,1,p=prob1)))
            k=k.flatten()
            #print(k)
            k=k.item(0)
            #print(k)
            #print("psition",k)
            #start = k.reshape(1)
            #return np.argmax(self.rewards.data)
            return k
            '''if self.epsilon>0.001:
                self.epsilon=self.epsilon*0.001
            else:
                self.epsilon=self.epsilon*1
            if np.random.uniform() < self.epsilon:
                # choose best action
                
                # some actions may have the same value, randomly choose on in these actions
                k= np.random.choice(np.argmax(keep))
            else:
                # choose random action
                k = np.random.choice((keep.size)-1)
                #= np.random.choice(np.argmin(keep))'''
                
           ''' keep=self.rewards.data.flatten()
            
            
            prio=keep/sum(keep)
            prob1=prio**self.alpha
            prob1=prob1/sum(prob1)
            k=np.random.choice(len(self.rewards),1,p=prob1)
            k=k.flatten()
            k=k.item(0)
            #print(k)
            return k'''
    
    
    def update_probs(self, idxes, probs):
        #assert len(idxes) == len(probs)
        #self.probs[idxes] = probs
        for idx, prob in zip(idxes, probs):
            assert prob > 0
            assert 0 <= idx < self.nb_entries
            self.probs[idx] = prob
            

    def update_td_errors(self, idxes, td_errors):
        #assert len(idxes) == len(td_errors)
        #self.td_errors[idxes] = self.fastsigmoid(td_errors)
        for idx, td_error in zip(idxes, td_errors):
            #assert  td_error >= 0
            assert 0 <= idx < self.nb_entries
            self.td_errors[idx] = sigmoid(td_error)
            

    def set_accu_rewards(self, nb_rollout, accu_r):
        # set the accumulative rewards for the last n_rollout transitioins
        start = self.next_idx - nb_rollout
        for idx in range(start, self.next_idx):
            self.accu_rewards[idx] = accu_r

    # randomly sample a batch for meta learning
    def meta_sample(self, batch_size):
        #batch_idxs = np.array(random.sample(range(self.nb_entries), batch_size))
        batch_idxs = np.random.random_integers(self.nb_entries-1, size=batch_size)
        X = self.get_X_by_idxs(batch_idxs)
        return batch_idxs, X

    # Extract features given indexes
    # Two features for now, TD-error and time feature
    def get_X_by_idxs(self, idxs):

        # Extract useful features
        #td_errors_batch = [self.td_errors[i] for i in idxs]
        #transition_rewards_batch = [self.transition_rewards[i] for i in idxs]
        #accu_rewards_batch = [self.accu_rewards[i] / self.current_accu_reward for i in idxs]

        ## Numpy Version
        td_errors_batch = self.td_errors[idxs]
        transition_rewards_batch = self.transition_rewards[idxs]
        accu_rewards_batch = self.accu_rewards[idxs]

        # Extract time feature
        time_batch = []
        for idx in idxs:
            # Extract time feature
            distance = (self.next_idx - idx) % self.limit
            time_feature = float(distance) / float(self.nb_entries)
            time_batch.append(time_feature)
        X = list(zip(td_errors_batch, transition_rewards_batch, accu_rewards_batch, time_batch))
        return np.array(X)

    @property
    def nb_entries(self):
        return len(self.observations0)

    @property
    def next_idx(self):
        return self.observations0._next_idx
    

                         
    def init_fastsigmoid(self):
        table = np.zeros(self.table_size)
        for i in range(self.table_size):
            x = (i * 2.0 * self.max_sigmoid / self.table_size) - self.max_sigmoid
            table[i] = 1.0 / (1.0 + math.exp(-x))
        return table
            
    def fastsigmoid(self, x):
        if type(x) is not np.ndarray:
            x  = np.array([x])
        output = np.zeros(x.shape)
        val = np.round(x, 3)
        max_ = np.full(x.shape, self.max_sigmoid) 
        min_ = np.full(x.shape, -self.max_sigmoid) 
        larger_than_max = np.greater(val, max_)
        smaller_than_min = np.less(val, min_)

        l_idx = np.argwhere(larger_than_max == True).flatten()
        sig_idx = np.intersect1d(np.argwhere(larger_than_max == False).flatten(), np.argwhere(smaller_than_min == False).flatten())
        inp_idx = (val[sig_idx] + self.max_sigmoid) * self.table_size / self.max_sigmoid / 2
        inp_idx = np.clip(inp_idx, 0, 999)
        #print(inp_idx.astype(int), sig_idx.shape, self.sigmoid[inp_idx.astype(int)])
        output[sig_idx] = self.sigmoid[inp_idx.astype(int)]
        output[l_idx] = 1.0
        return output

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 0.5
  



'''import tensorflow as tf
print(tf.__version__)
print('hello world')'''

