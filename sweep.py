from absl import app, flags
import os
from enum import Enum
import copy

import jax
import numpy as np

from brax.io import html

from src.envs import unitree_go2_energy_test as unitree_go2
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


def cost(phase, target):
    # Load from Env:
    velocity_target = 0.375
    phase_targets = [0.25, 0.5, 0.75]

    env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_mjx.xml',
    )
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        restore_iteration=FLAGS.checkpoint_iteration,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Initialize Simulation:
    key = jax.random.key(0)

    gaits = []
    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)

    num_steps = 1000
    steady_state_ratio = 0.8
    states = []
    first_contact = []
    contacts = []
    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        action, _ = inference_fn(state.obs, subkey)
        state = step_fn(state, action)
        states.append(state.pipeline_state)

        # Get Steady State:
        steady_state_condition = (
            (i > int((1.0-steady_state_ratio) * num_steps))
            & (i <= int(steady_state_ratio * num_steps))
        )
        if steady_state_condition:
            first_contact.append(state.info['first_contact'])
            contacts.append(state.info['previous_contact'])

    first_contact = np.asarray(first_contact)
    contacts = np.asarray(contacts)

    # Find Start of Stance Indicies for Dominant Foot: [Right Front]
    indicies = np.where(first_contact[:, Feet.front_right.value])[0]
    stride_lengths = np.diff(indicies)

    front_left = []
    hind_right = []
    hind_left = []
    for i in indicies:
        front_left.append(
            np.where(first_contact[i:, Feet.front_left.value])[0][0]
        )
        hind_right.append(
            np.where(first_contact[i:, Feet.hind_right.value])[0][0]
        )
        hind_left.append(
            np.where(first_contact[i:, Feet.hind_left.value])[0][0]
        )

    # Calculate Phases:
    front_left = np.asarray(front_left)[:-1]
    hind_right = np.asarray(hind_right)[:-1]
    hind_left = np.asarray(hind_left)[:-1]

    front_left_phase = front_left / stride_lengths
    hind_right_phase = hind_right / stride_lengths
    hind_left_phase = hind_left / stride_lengths

    # # Account for phase offset: Subtract 1.0
    front_left_phase, hind_right_phase, hind_left_phase = list(
        map(lambda x: np.where(x > 1.0, x - 1.0, x), [front_left_phase, hind_right_phase, hind_left_phase]),
    )

    # Account for phase offset: Modulus
    # front_left_phase, hind_right_phase, hind_left_phase = list(
    #     map(lambda x: np.where(x > 1.0, x % 1, x), [front_left_phase, hind_right_phase, hind_left_phase]),
    # )

    avg_front_left_phase, avg_hind_right_phase, avg_hind_left_phase = list(
        map(lambda x: np.mean(x), [front_left_phase, hind_right_phase, hind_left_phase]),
    )

    # Optimistic Cost: Find closest phase to target:
    avg_phases = [avg_front_left_phase, avg_hind_right_phase, avg_hind_left_phase]
    error = np.sort(avg_phases) - np.asarray(phase_targets)
    cost = np.sum(np.square(error))


from absl import app, flags
import os
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import optax

import wandb
import orbax.checkpoint as ocp

from src.envs import unitree_go2_energy_test as unitree_go2
from src.algorithms.ppo import network_utilities as ppo_networks
from src.algorithms.ppo.loss_utilities import loss_function
from src.distribution_utilities import ParametricDistribution
from src.algorithms.ppo.train import train
from src.algorithms.ppo import checkpoint_utilities
from src.algorithms.ppo.load_utilities import load_checkpoint

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

jax.config.update("jax_enable_x64", True)
wandb.require('core')

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)
flags.DEFINE_string(
    'tag', '', 'Tag for wandb run.', short_name='t',
)


def objective(make_policy, params):
    class Feet(Enum):
        front_left = 0
        hind_left = 1
        front_right = 2
        hind_right = 3

    # Load from Env:
    velocity_target = 0.375
    phase_targets = [0.25, 0.5, 0.75]

    env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_mjx.xml',
    )
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Make Inference Function:
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Initialize Simulation:
    key = jax.random.key(0)

    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)

    num_steps = 1000
    steady_state_ratio = 0.8
    states = []
    first_contact = []
    contacts = []
    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        action, _ = inference_fn(state.obs, subkey)
        state = step_fn(state, action)
        states.append(state.pipeline_state)

        # Get Steady State:
        steady_state_condition = (
            (i > int((1.0-steady_state_ratio) * num_steps))
            & (i <= int(steady_state_ratio * num_steps))
        )
        if steady_state_condition:
            first_contact.append(state.info['first_contact'])
            contacts.append(state.info['previous_contact'])

    first_contact = np.asarray(first_contact)
    contacts = np.asarray(contacts)

    # Find Start of Stance Indicies for Dominant Foot: [Right Front]
    indicies = np.where(first_contact[:, Feet.front_right.value])[0]
    stride_lengths = np.diff(indicies)

    front_left = []
    hind_right = []
    hind_left = []
    for i in indicies:
        front_left.append(
            np.where(first_contact[i:, Feet.front_left.value])[0][0]
        )
        hind_right.append(
            np.where(first_contact[i:, Feet.hind_right.value])[0][0]
        )
        hind_left.append(
            np.where(first_contact[i:, Feet.hind_left.value])[0][0]
        )

    # Calculate Phases:
    front_left = np.asarray(front_left)[:-1]
    hind_right = np.asarray(hind_right)[:-1]
    hind_left = np.asarray(hind_left)[:-1]

    front_left_phase = front_left / stride_lengths
    hind_right_phase = hind_right / stride_lengths
    hind_left_phase = hind_left / stride_lengths

    # # Account for phase offset: Subtract 1.0
    front_left_phase, hind_right_phase, hind_left_phase = list(
        map(lambda x: np.where(x > 1.0, x - 1.0, x), [front_left_phase, hind_right_phase, hind_left_phase]),
    )

    # Account for phase offset: Modulus
    # front_left_phase, hind_right_phase, hind_left_phase = list(
    #     map(lambda x: np.where(x > 1.0, x % 1, x), [front_left_phase, hind_right_phase, hind_left_phase]),
    # )

    avg_phases = list(
        map(lambda x: np.mean(x), [front_left_phase, hind_right_phase, hind_left_phase]),
    )

    # Sorted Cost Error:
    error = np.sort(avg_phases) - np.asarray(phase_targets)
    cost = np.sum(np.square(error))

    return cost

    

def main(argv=None):
    # Config:
    reward_config = unitree_go2.RewardConfig(
        tracking_forward_velocity=2.0,
        lateral_velocity=-1.0,
        angular_velocity=-1.0,
        mechanical_power=-2e-2,
        torque=-2e-3,
        termination=-1.0,
    )
    velocity_target = 0.375

    # Metadata:
    network_metadata = checkpoint_utilities.network_metadata(
        policy_layer_size=128,
        value_layer_size=256,
        policy_depth=4,
        value_depth=5,
        activation='nn.swish',
        kernel_init='jax.nn.initializers.lecun_uniform()',
        action_distribution='ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh())',
    )
    loss_metadata = checkpoint_utilities.loss_metadata(
        clip_coef=0.3,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.97,
        gae_lambda=0.95,
        normalize_advantages=True,
    )
    training_metadata = checkpoint_utilities.training_metadata(
        num_epochs=20,
        num_training_steps=20,
        episode_length=1000,
        num_policy_steps=25,
        action_repeat=1,
        num_envs=4096,
        num_evaluation_envs=128,
        num_evaluations=1,
        deterministic_evaluation=True,
        reset_per_epoch=False,
        seed=0,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
        optimizer='optax.adam(3e-4)',
    )

    # Start Wandb and save metadata:
    run = wandb.init(
        project='gait_analysis',
        group='unitree_go2',
        tags=[FLAGS.tag],
        config={
            'reward_config': reward_config,
            'network_metadata': network_metadata,
            'loss_metadata': loss_metadata,
            'training_metadata': training_metadata,
        },
    )

    # Initialize Functions with Params:
    randomization_fn = unitree_go2.domain_randomize
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_layer_sizes=(network_metadata.policy_layer_size, ) * network_metadata.policy_depth,
        value_layer_sizes=(network_metadata.value_layer_size, ) * network_metadata.value_depth,
        activation=nn.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    loss_fn = functools.partial(
        loss_function,
        clip_coef=loss_metadata.clip_coef,
        value_coef=loss_metadata.value_coef,
        entropy_coef=loss_metadata.entropy_coef,
        gamma=loss_metadata.gamma,
        gae_lambda=loss_metadata.gae_lambda,
        normalize_advantages=loss_metadata.normalize_advantages,
    )
    env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_barkour_hfield_mjx.xml',
        config=reward_config,
    )
    eval_env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_barkour_hfield_mjx.xml',
        config=reward_config,
    )
    render_env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_barkour_hfield_mjx.xml',
        config=reward_config,
    )

    def progress_fn(iteration, num_steps, metrics):
        print(
            f'Iteration: {iteration} \t'
            f'Num Steps: {num_steps} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if num_steps > 0:
            print(
                f'Training Loss: {metrics["training/loss"]:.3f} \t'
                f'Policy Loss: {metrics["training/policy_loss"]:.3f} \t'
                f'Value Loss: {metrics["training/value_loss"]:.3f} \t'
                f'Entropy Loss: {metrics["training/entropy_loss"]:.3f} \t'
                f'Training Wall Time: {metrics["training/walltime"]:.3f} \t'
            )
        print('\n')

    # Setup Checkpoint Manager:
    manager_options = checkpoint_utilities.default_checkpoint_options()
    checkpoint_direrctory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{run.name}",
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=manager_options,
        item_names=(
            'train_state',
            'network_metadata',
            'loss_metadata',
            'training_metadata',
        ),
    )
    checkpoint_fn = functools.partial(
        checkpoint_utilities.save_checkpoint,
        manager=manager,
        network_metadata=network_metadata,
        loss_metadata=loss_metadata,
        training_metadata=training_metadata,
    )

    train_fn = functools.partial(
        train,
        num_epochs=training_metadata.num_epochs,
        num_training_steps=training_metadata.num_training_steps,
        episode_length=training_metadata.episode_length,
        num_policy_steps=training_metadata.num_policy_steps,
        action_repeat=training_metadata.action_repeat,
        num_envs=training_metadata.num_envs,
        num_evaluation_envs=training_metadata.num_evaluation_envs,
        num_evaluations=training_metadata.num_evaluations,
        deterministic_evaluation=training_metadata.deterministic_evaluation,
        reset_per_epoch=training_metadata.reset_per_epoch,
        seed=training_metadata.seed,
        batch_size=training_metadata.batch_size,
        num_minibatches=training_metadata.num_minibatches,
        num_ppo_iterations=training_metadata.num_ppo_iterations,
        normalize_observations=training_metadata.normalize_observations,
        network_factory=make_networks_factory,
        optimizer=optax.adam(3e-4),
        loss_function=loss_fn,
        progress_fn=progress_fn,
        randomization_fn=randomization_fn,
        checkpoint_fn=checkpoint_fn,
        wandb=run,
        render_environment=render_env,
        render_interval=5,
    )

    policy_generator, params, metrics = train_fn(
        environment=env,
        evaluation_environment=eval_env,
    )

    run.finish()


if __name__ == '__main__':
    app.run(main)
