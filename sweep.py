from typing import List, Callable
from enum import Enum
from absl import app, flags
import os
from pathlib import Path
import yaml
import functools

import jax
import flax.linen as nn
import distrax
import optax

import numpy as np

import wandb
import orbax.checkpoint as ocp

from brax.envs.base import PipelineEnv

from src.envs import unitree_go2_energy_test as unitree_go2
from src.algorithms.ppo import network_utilities as ppo_networks
from src.algorithms.ppo.loss_utilities import loss_function
from src.distribution_utilities import ParametricDistribution
from src.algorithms.ppo.train import train
from src.algorithms.ppo import checkpoint_utilities
from src.algorithms.ppo.load_utilities import load_checkpoint
from src.algorithms.ppo.network_utilities import PPONetworkParams

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

jax.config.update("jax_enable_x64", True)
wandb.require('core')


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
    
    # Phase Target + Missed Stride + Termination Cost:
    cost = phase_weight * np.sum(np.square(phase_error)) + period_weight * np.sum(period_error) + termination_weight * termination_error

    # Cost Ideas: 
    # Phase Target + Velocity Target
    # Phase Target + Velocity Target + Stride Frequency

    return cost


def sweep_main(argv=None):
    # Setup Wandb:
    run = wandb.init(
        project='gait_analysis',
        group='parameter_sweep',
    )

    # Gait Parameters:
    velocity_target = 0.375
    phase_targets = [0.25, 0.5, 0.75]

    # Config:
    reward_config = unitree_go2.RewardConfig(
        tracking_forward_velocity=2.0,
        lateral_velocity=run.config.lateral_velocity,
        angular_velocity=run.config.angular_velocity,
        mechanical_power=run.config.mechanical_power,
        torque=run.config.torque,
        termination=-1.0,
    )

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

    # Log Metadata:
    run.config.update({
        'reward_config': reward_config,
        'network_metadata': network_metadata,
        'loss_metadata': loss_metadata,
        'training_metadata': training_metadata,
    })

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
    )

    make_policy, params, metrics = train_fn(
        environment=env,
        evaluation_environment=eval_env,
    )

    # Outer Objective Function:
    objective_env = unitree_go2.UnitreeGo2Env(
        velocity_target=velocity_target,
        filename='unitree_go2/scene_mjx.xml',
    )

    cost = objective(
        velocity_target=velocity_target,
        phase_targets=phase_targets,
        env=objective_env ,
        make_policy=make_policy,
        params=params,
    )

    wandb.log({'score': cost})

    run.finish()


def main(argv=None):
    config = yaml.safe_load(Path('sweep_config.yaml').read_text())
    sweep_id = wandb.sweep(sweep=config, project='gait_analysis')
    wandb.agent(sweep_id, function=sweep_main)


if __name__ == '__main__':
    app.run(main)
