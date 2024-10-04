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

from src.envs import unitree_go2_energy as unitree_go2
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
flags.DEFINE_bool(
    'overwrite_metadata', False, 'Overwrite metadata from restored checkpoint.', short_name='o',
)
flags.DEFINE_string(
    'tag', '', 'Tag for wandb run.', short_name='t',
)
flags.DEFINE_bool(
    'reinitialize_policy_std', False, 'Reinitialize Policy Output Layer parameters that correlate to Action STD.', short_name='r',
)


def main(argv=None):
    # Config:
    reward_config = unitree_go2.RewardConfig()
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
        num_epochs=50,
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

    restored_checkpoint = None
    if FLAGS.checkpoint_name is not None:
        restored_checkpoint, metadata = load_checkpoint(
            checkpoint_name=FLAGS.checkpoint_name,
            environment=env,
            restore_iteration=FLAGS.checkpoint_iteration,
        )

        if FLAGS.reinitialize_policy_std:
            # Reset Policy std params to enable exploration:
            key = jax.random.PRNGKey(42)
            key, bias_key, kernel_key = jax.random.split(key, num=3)
            policy_params = restored_checkpoint.train_state.params.policy_params
            output_params = policy_params['params'][f'dense_{metadata.network_metadata.policy_depth}']
            params = jax.tree.map(lambda x: jnp.split(x, 2, axis=-1), output_params)
            bias_shape = jnp.expand_dims(params['bias'][-1], axis=0).shape
            kernel_shape = params['kernel'][-1].shape
            bias = jnp.concatenate(
                [
                    params['bias'][0],
                    jax.nn.initializers.lecun_uniform()(
                        key=bias_key, shape=bias_shape,
                    ).flatten(),
                ],
                axis=-1,
                dtype=jnp.float32,
            )
            kernel = jnp.concatenate(
                [
                    params['kernel'][0],
                    jax.nn.initializers.lecun_uniform()(
                        key=kernel_key, shape=kernel_shape,
                    ),
                ],
                axis=-1,
                dtype=jnp.float32,
            )
            output_layer = {
                'bias': bias,
                'kernel': kernel,
            }
            policy_params['params'].update({f'dense_{metadata.network_metadata.policy_depth}': output_layer})
            restored_checkpoint.train_state.params.replace(policy_params=policy_params)

        if FLAGS.overwrite_metadata:
            run.config.update({
                'network_metadata': metadata.network_metadata,
                'loss_metadata': metadata.loss_metadata,
                'training_metadata': metadata.training_metadata,
            })

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
        restored_checkpoint=restored_checkpoint,
    )

    policy_generator, params, metrics = train_fn(
        environment=env,
        evaluation_environment=eval_env,
    )

    run.finish()


if __name__ == '__main__':
    app.run(main)
