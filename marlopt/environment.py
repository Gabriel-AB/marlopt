from optfuncs.core import Function
from pettingzoo import ParallelEnv
from gym.spaces import Box
import numpy as np
import tensorflow as tf


class OptFuncParallelEnv(ParallelEnv):
  """
  Parallel Environment for Global Optimization

  Simulates agents trying to optimize a optimization function

  This env is partially observable and make uses the following 
  observations for each agent:
    (agent_action)
    (best_agent_reward - agent_reward)
    (best_agent_state - agent_state)
  """

  metadata = {'render.modes': ['human'], 'name': 'OptFuncParallelEnv'}

  def __init__(self,
               function: Function,
               dims: int,
               num_agents: int,
               max_steps: int = 3000):
    """
    function: An optimization function
    dims: Dimentions of optimization
    """

    self.func = function
    self.dims = dims
    self.possible_agents = [f'agent_{i}' for i in range(num_agents)]
    self.max_steps = max_steps
    
    self.observation_spaces = {
      agent: Box(-np.inf, np.inf, (2*dims + 1,), dtype=float)
      for agent in self.possible_agents
    }
    self.action_spaces = {
      agent: Box(-np.inf, np.inf, (dims,), float)
      for agent in self.possible_agents
    }
    self.gen = tf.random.Generator.from_non_deterministic_state()

  
  def observation_space(self, agent):
    return self.observation_spaces[agent]

  def action_space(self, agent):
    return self.action_spaces[agent]
  
  def update_observations(self, actions):
    best_agent = max(self.rewards)

    for agent in self.observations:
      concat = [
        actions[agent],
        [self.rewards[best_agent] - self.rewards[agent]],
        self.states[best_agent] - self.states[agent]
      ]
      
      self.observations[agent] = tf.concat(concat, axis=0)
  
  def seed(self, seed):
    self.gen.reset_from_seed(seed)

  def reset(self):
    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0.0 for agent in self.agents}
    self._cumulative_rewards = {agent: 0 for agent in self.agents}
    self.dones = {agent: False for agent in self.agents}
    self.infos = {agent: {} for agent in self.agents}
    self.states = {
      agent: self.gen.uniform(
        (self.dims,), 
        *self.func.domain,
        dtype=tf.float64,
        name=agent
      ) for agent in self.agents
    }

    # Observation
    self.observations = {agent: None for agent in self.agents}
    actions = {
      agent: tf.zeros(
        (self.dims,),
        dtype=tf.float64,
      ) for agent in self.agents
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
