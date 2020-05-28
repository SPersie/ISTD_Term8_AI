
# coding: utf-8

# In[1]:


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# In[2]:


env = gym.make('LunarLander-v2').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


class ReplayMemory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def memory(self):
    	return self.memory

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# In[4]:


class reinforce(nn.Module):

    def __init__(self, outputs):
        super(reinforce, self).__init__()
        self.fc1 = nn.Linear(8, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, outputs)
        self.sm = nn.Softmax()
        self.fc4 = nn.Linear(24, 1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x_fc1 = F.leaky_relu(self.fc1(x))
        x_fc2 = F.leaky_relu(self.fc2(x_fc1))
        pi_opt = self.sm(self.fc3(x_fc2)).clamp(0.001, 1.0)
        v_opt = self.fc4(x_fc2)
        return pi_opt, v_opt


# In[5]:


def select_action(state):
    m = torch.distributions.Categorical(torch.tensor([0.25, 0.75]))
    m_type = m.sample().data.tolist()
    if m_type == 0:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return nn(state)[0].max(1)[1].view(1, 1)


# In[6]:


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        episode_mean.append(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# In[7]:


def optimzer_model():
    # get memeory
    transitions = memory.memory
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
#     print(state_batch.shape)
    action_batch = torch.cat(batch.action)
    # print(len(batch.reward))
    temp = []
    for i in range(len(batch.reward)):
        temp.append(batch.reward[i].type(torch.FloatTensor))
    reward_batch = torch.cat(tuple(temp))
    
    g_list = reward_batch.data.tolist()
    for i in reversed(range(1, len(g_list))):
        g_list[i-1] = g_list[i-1] + 0.999 * g_list[i]
    g_tensor = torch.FloatTensor(g_list).view(-1,1)
#     print(g_tensor.shape)
    
    state_action_values = nn(state_batch)[0].gather(1, action_batch)
    b_tensor = nn(state_batch)[1].detach().view(-1,1)
    l1 = (g_tensor - b_tensor) * torch.log(state_action_values)
#     print(l1.shape)
    l1 = torch.sum(l1)
    l2 = F.smooth_l1_loss(g_tensor, b_tensor)
#     print(l2)
    loss = l1 + l2
#     print(loss)
    optimizer.zero_grad()
    loss.backward()
    # for param in nn.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[8]:


env.reset()


# In[9]:


n_actions = env.action_space.n
nn = reinforce(n_actions)


# In[12]:


num_episodes = 500
episode_durations = []
episode_mean = []
memory = ReplayMemory()
optimizer = optim.RMSprop(nn.parameters(), lr=8 * math.exp(-3))

# print(num_episodes)
for i in range(num_episodes):
    # print(i)
    state = torch.FloatTensor([env.reset()])
    for t in count():
        # print(state)
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        # Observe new state
        if not done:
            next_state = torch.FloatTensor([next_state])
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        # optimize_model()
        if done:
            episode_durations.append(min(500, t + 1))
            plot_durations()
            break
    optimzer_model()
env.render()
env.close()
plt.ioff()
plt.show()



