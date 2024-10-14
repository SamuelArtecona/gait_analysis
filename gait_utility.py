from typing import List, Callable
from absl import app, flags
import os
from enum import Enum
import copy

import jax
import numpy as np


from brax.io import html
from brax.envs.base import PipelineEnv

from src.envs import unitree_go2_energy_test as unitree_go2
from src.algorithms.ppo.network_utilities import PPONetworkParams
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


class Feet(Enum):
    front_left = 0
    hind_left = 1
    front_right = 2
    hind_right = 3


def objective(
    velocity_target: float,
    phase_targets: List[float],
    env: PipelineEnv,
    make_policy: Callable,
    params: PPONetworkParams,
    num_steps: int = 1000,
    ratio: float = 0.8,
) -> float:
    class Feet(Enum):
        front_left = 0
        hind_left = 1
        front_right = 2
        hind_right = 3

    # Create Environment:
    reset_fn = jax.jit(env.reset)

    # Make Inference Function:
    inference_fn = make_policy(params)

    # Initialize Simulation:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)

    def loop(carry, xs):
        key, state = carry
        key, subkey = jax.random.split(key)
        action, _ = inference_fn(state.obs, subkey)
        state = env.step(state, action)
        data = (state.pipeline_state, state.info['first_contact'], state.done)
        return (key, state), data

    _, (states, first_contact, done) = jax.lax.scan(
        f=loop,
        init=(key, state),
        xs=(),
        length=num_steps,
    )

    # Slice Data:
    start_slice, end_slice = int((1-ratio) * num_steps), int(ratio * num_steps)
    first_contact = np.asarray(
        first_contact,
    )[start_slice:end_slice]

    # Find Start of Stance Indicies for Dominant Foot: [Right Front]
    indicies = np.where(first_contact[:, Feet.front_right.value])[0]
    stride_lengths = np.diff(indicies)

    front_left = []
    hind_right = []
    hind_left = []
    for i in indicies:
        lengths = list(
            map(
                lambda x: x[0] if np.size(x) > 0 else 0,
                [
                    np.where(first_contact[i:, Feet.front_left.value])[0],
                    np.where(first_contact[i:, Feet.hind_right.value])[0],
                    np.where(first_contact[i:, Feet.hind_left.value])[0],
                ],
            )
        )

        front_left.append(lengths[0])
        hind_right.append(lengths[1])
        hind_left.append(lengths[2])

    # Calculate array lengths:
    front_left = np.asarray(front_left)
    hind_right = np.asarray(hind_right)
    hind_left = np.asarray(hind_left)
    front_left_length = front_left.shape[0]
    hind_right_length = hind_right.shape[0]
    hind_left_length = hind_left.shape[0]

    array_lengths = np.minimum(
        [front_left_length, hind_right_length, hind_left_length],
        stride_lengths.shape[0],
    )

    front_left = front_left[:array_lengths[0]]
    hind_right = hind_right[:array_lengths[1]]
    hind_left = hind_left[:array_lengths[2]]

    # Calculate Phases:
    front_left_phase = front_left / stride_lengths
    hind_right_phase = hind_right / stride_lengths
    hind_left_phase = hind_left / stride_lengths

    # Account for phase offset: Subtract 1.0
    # front_left_phase, hind_right_phase, hind_left_phase = list(
    #     map(
    #         lambda x: np.where(x > 1.0, x - 1.0, x),
    #         [front_left_phase, hind_right_phase, hind_left_phase],
    #     ),
    # )

    # Account for phase offset: Modulus
    front_left_phase, hind_right_phase, hind_left_phase = list(
        map(lambda x: np.where(x > 1.0, x % 1, x), [front_left_phase, hind_right_phase, hind_left_phase]),
    )

    avg_phases = list(
        map(
            lambda x: np.mean(x),
            [front_left_phase, hind_right_phase, hind_left_phase],
        ),
    )

    # Sorted Cost Error:
    phase_weight = 1.0
    phase_error = np.sort(avg_phases) - np.asarray(phase_targets)

    # Missed Stride Cost:
    period_weight = 1.0
    period_error = np.where(array_lengths == stride_lengths.shape[0], 0.0, 1.0)

    # Failed Locomotion Cost:
    termination_weight = 100.0
    termination_error = np.any(done)
    
    # Phase Target Cost:
    cost = phase_weight * np.sum(np.square(phase_error)) + period_weight * np.sum(period_error) + termination_weight * termination_error

    # Cost Ideas: 
    # Phase Target + Velocity Target
    # Phase Target + Velocity Target + Stride Frequency

    return cost


def main(argv=None):
    # Load from Env:
    velocity_target = 0.375
    phase_targets = [0.25, 0.5, 0.75]

    env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_mjx.xml',
    )

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        restore_iteration=FLAGS.checkpoint_iteration,
        environment=env,
    )

    # Run Objective Function:
    cost = objective(
        velocity_target=velocity_target,
        phase_targets=phase_targets,
        env=env,
        make_policy=make_policy,
        params=params,
    )

    print(f'Cost: {cost:.3f}')


if __name__ == '__main__':
    app.run(main)
