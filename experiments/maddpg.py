# %% Imports
import functools
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
from mava.utils.enums import ArchitectureType
import numpy as np
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from mava.components.tf import networks
from acme.tf import utils as tf2_utils


from mava import specs as mava_specs
from mava.systems.tf import maddpg
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.wrappers import MonitorParallelEnvironmentLoop
from mava.components.tf import architectures
from mava.utils.loggers import logger_utils

# %% Importing Optimization environment
from marlopt.environments import optfunc_v0
import optfuncs.tensorflow_functions as tff
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper


# %% Config

env_name = "Sphere" # @param ["Sphere", "Ackley", "Levy", "Rosenbrock"]
dims = 2 # @param {type: 'integer'}
num_agents = 1 # @param {type: 'integer'}



def make_environment(env_name: str, dims: int, num_agents: int, *args, **kwargs):
    function = getattr(tff, env_name)()
    env = optfunc_v0.parallel_env(function, dims=dims, num_agents=num_agents)
    env = PettingZooParallelEnvWrapper(env)
    return env


environment_factory = functools.partial(
    make_environment,
    env_name=env_name,
    dims=dims,
    num_agents=num_agents
)

network_factory = lp_utils.partial_kwargs(
    maddpg.make_default_networks, 
    archecture_type=ArchitectureType.feedforward
)

# %% Config log
# Directory to store checkpoints and log data. 
base_dir = "mava"

# File name 
mava_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Log every [log_every] seconds
log_every = 15
logger_factory = functools.partial(
    logger_utils.make_logger,
    directory=base_dir,
    to_terminal=True,
    to_tensorboard=True,
    time_stamp=mava_id,
    time_delta=log_every,
)

# Checkpointer appends "Checkpoints" to checkpoint_dir
checkpoint_dir = f"{base_dir}/{mava_id}"

# %% Creating MADDPG system
system = maddpg.MADDPG(
    environment_factory=environment_factory,
    network_factory=network_factory,
    logger_factory=logger_factory,
    num_executors=1,
    policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
    critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
    checkpoint_subpath=checkpoint_dir,
    max_gradient_norm=40.0,
    checkpoint=False,
    batch_size=1024,

    # Record agents in environment. 
    eval_loop_fn=MonitorParallelEnvironmentLoop,
    eval_loop_fn_kwargs={"path": checkpoint_dir, "record_every": 10, "fps": 5},
).build()

# %% Running
local_resources = lp_utils.to_device(program_nodes=system.groups.keys(),nodes_on_gpu=["trainer"])

lp.launch(
    system,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    # terminal="output_to_files",
    terminal="output_to_files",
    local_resources=local_resources,
)
# %%
!cat /tmp/launchpad_out/evaluator/0

# %%
!cat /tmp/launchpad_out/trainer/0


# %%
# import tensorboard as tb