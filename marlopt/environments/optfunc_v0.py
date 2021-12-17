from optfuncs import core
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel
from gym.spaces import Box
import numpy as np

class DummyFunction(core.Function):
  def __init__(self, domain = core.Domain(-1.0, 1.0)):
      super().__init__(domain)
  def __call__(self, x):
      return sum(x)

def env():
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env():
    env = parallel_env(DummyFunction(), 2, 3)
    env = from_parallel(env)
    return env


class parallel_env(ParallelEnv):
  """
  Environment for Global Optimization with Partial Observability

  Simulates agents trying to optimize a mathematical optimization function

  This env is partially observable and make uses the following 
  observation structure for each agent:
    (agent_action)
    (best_agent_reward - agent_reward)
    (best_agent_state - agent_state)
  """

  metadata = {'render.modes': ['human'], 'name': 'OptFuncParallelEnv'}

  def __init__(self,
               function: core.Function = DummyFunction(),
               dims: int = 2,
               num_agents: int = 3,
               max_steps: int = 256):
    """
    function: An optimization function
    dims: Dimentions of optimization
    """

    self.func = function
    self.dims = dims
    self.possible_agents = [f'agent_{i}' for i in range(num_agents)]
    self.max_steps = max_steps
    
    lower_bound = np.array(2*dims*[function.domain.min] + [-np.inf])
    upper_bound = np.array(2*dims*[function.domain.max] + [ np.inf])

    self.observation_spaces = {
      agent: Box(lower_bound, upper_bound, (2*dims + 1,), dtype=np.float32)
      for agent in self.possible_agents
    }
    self.action_spaces = {
      agent: Box(-np.inf, np.inf, (dims,), np.float32)
      for agent in self.possible_agents
    }
  
  def observation_space(self, agent):
    return self.observation_spaces[agent]

  def action_space(self, agent):
    return self.action_spaces[agent]
  
  def update_observations(self, actions):
    best_agent = max(self.rewards, key=self.rewards.get)

    for agent in self.agents:
      concat = [
        actions[agent],
        self.states[best_agent] - self.states[agent],
        [self.rewards[best_agent] - self.rewards[agent]],
      ]
      self.observations[agent] = np.concatenate(concat, axis=0, dtype=np.float32)
  
  def reset(self):
    self.agents = self.possible_agents[:]
    self._cumulative_rewards = {agent: 0 for agent in self.agents}
    self.dones = {agent: False for agent in self.agents}
    self.infos = {agent: {} for agent in self.agents}
    self.states = {
      agent: np.random.uniform(*self.func.domain, (self.dims,)).astype(np.float32)
      for agent in self.agents
    }

    # Observation
    self.rewards = {agent: -self.func(self.states[agent]) for agent in self.agents}
    self.observations = {}
    actions = {
      agent: box.sample() for agent, box in self.action_spaces.items()
    }

    self.update_observations(actions)
    self.num_moves = 0
    return self.observations


  def step(self, actions):
    
    if not actions:
      return {}, {}, {}, {}
    
    self.num_moves += 1
    end = self.num_moves > self.max_steps

    for agent in actions:
      next_state = self.states[agent] + actions[agent]
      self.rewards[agent] = -self.func(next_state)
      self.states[agent] = next_state

      # agent death param
      self.dones[agent] = end
      # if end: self.agents.remove(agent)

    self.update_observations(actions)

    return self.observations, self.rewards, self.dones, self.infos
