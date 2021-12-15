import launchpad as lp
from launchpad.nodes.python import local_multi_processing

from mava.systems.tf import maddpg
from mava.utils import lp_utils
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper

from marlopt.environment import OptFuncParallelEnv
from optfuncs.tensorflow_functions import Sphere

def environment_factory(evaluation: bool):
  del evaluation
  env = OptFuncParallelEnv(Sphere(), dims=2, num_agents=1)
  return PettingZooParallelEnvWrapper(env)

def main():
  program = maddpg.MADDPG(
    environment_factory=environment_factory,
    network_factory=lp_utils.partial_kwargs(maddpg.make_default_networks),
  ).build()

  
  env_vars = {"CUDA_VISIBLE_DEVICES": 0}
  local_resources = {
    "trainer": local_multi_processing.PythonProcess(env=env_vars),
    "evaluator": local_multi_processing.PythonProcess(env=env_vars),
    "executor": local_multi_processing.PythonProcess(env=env_vars),
  }

  # Launch.
  lp.launch(
    program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal="current_terminal",
  )


if __name__ == '__main__':
  main()



