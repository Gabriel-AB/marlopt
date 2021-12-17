import numpy as np
import unittest

from marlopt.environments import optfunc_v0
from pettingzoo.test import parallel_api_test


class TestOptFuncEnv_v0(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    self.env = optfunc_v0.parallel_env()
    self.env.reset()

  def test_api(self):
    parallel_api_test(self.env)

  def test_dtype(self):
    obs = self.env.reset()
    for o in obs.values():
      self.assertEqual(o.dtype, np.float32)
  
  def test_observation(self):
    env = self.env
    obs = env.reset()
    best_agent = max(env.rewards, key=env.rewards.get)
    
    # Checking agent observations which are composed of
    # [actions, agent_state - best_agent_state, agent_reward - best_agent_reward]

    # obs: since actions here are randomic, we will not use them.
    for agent in env.agents:
      concat = np.concatenate([
        env.states[best_agent] - env.states[agent], 
        [env.rewards[best_agent] - env.rewards[agent]]
      ], axis=0, dtype=np.float32)
      self.assertTrue(all(obs[agent][2:] == concat))

if __name__ == '__main__':
  unittest.main()
