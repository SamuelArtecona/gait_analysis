from typing import Any, Optional, Union

import jax.numpy as jnp

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
import optax
import flax.struct
from brax.training.acme import running_statistics
import src.module_types as types
import src.distribution_utilities as distribution
import src.algorithms.ppo.network_utilities as ppo_networks


@flax.struct.dataclass
class TrainState:
    opt_state: optax.OptState
    params: ppo_networks.PPONetworkParams
    normalization_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


@flax.struct.dataclass
class RestoredCheckpoint:
    network: ppo_networks.PPONetworks
    train_state: TrainState


@flax.struct.dataclass
class network_metadata:
    policy_layer_size: int
    value_layer_size: int
    policy_depth: int
    value_depth: int
    activation: Union[types.ActivationFn, str]
    kernel_init: Union[types.Initializer, str]
    action_distribution: Union[distribution.ParametricDistribution, str]


@flax.struct.dataclass
class loss_metadata:
    clip_coef: float
    value_coef: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    normalize_advantages: bool


@flax.struct.dataclass
class training_metadata:
    num_epochs: int
    num_training_steps: int
    episode_length: int
    num_policy_steps: int
    action_repeat: int
    num_envs: int
    num_evaluation_envs: int
    num_evaluations: int
    deterministic_evaluation: bool
    reset_per_epoch: bool
    seed: int
    batch_size: int
    num_minibatches: int
    num_ppo_iterations: int
    normalize_observations: bool
    optimizer: Union[optax.GradientTransformation, str]


def empty_network_metadata() -> network_metadata:
    return network_metadata(
        policy_layer_size=0,
        value_layer_size=0,
        policy_depth=0,
        value_depth=0,
        activation='',
        kernel_init='',
        action_distribution='',
    )


def empty_loss_metadata() -> loss_metadata:
    return loss_metadata(
        clip_coef=0.0,
        value_coef=0.0,
        entropy_coef=0.0,
        gamma=0.0,
        gae_lambda=0.0,
        normalize_advantages=False,
    )


def empty_training_metadata() -> training_metadata:
    return training_metadata(
        num_epochs=0,
        num_training_steps=0,
        episode_length=0,
        num_policy_steps=0,
        action_repeat=0,
        num_envs=0,
        num_evaluation_envs=0,
        num_evaluations=0,
        deterministic_evaluation=False,
        reset_per_epoch=False,
        seed=0,
        batch_size=0,
        num_minibatches=0,
        num_ppo_iterations=0,
        normalize_observations=False,
        optimizer='',
    )


def default_checkpoint_options() -> CheckpointManagerOptions:
    options = CheckpointManagerOptions(
        max_to_keep=10,
        save_interval_steps=1,
        create=True,
    )
    return options


def default_checkpoint_metadata() -> dict:
    return {'iteration': 0}


def save_checkpoint(
    iteration: int,
    manager: CheckpointManager,
    train_state: TrainState,
    **metadata: Union[dict[str, Any], flax.struct.PyTreeNode],
) -> None:
    # Save Checkpoint:
    args = {
        'train_state': ocp.args.PyTreeSave(train_state),  # type: ignore
    }
    metadata = {
        key: ocp.args.PyTreeSave(value) for key, value in metadata.items()  # type: ignore
    }
    args.update(metadata)  # type: ignore
    manager.save(
        iteration,
        args=ocp.args.Composite(
            **args,
        ),
    )


def load_checkpoint(
    manager: CheckpointManager,
    restore_iteration: Optional[int] = None,
    **data: Union[dict[str, Any], flax.struct.PyTreeNode],
) -> Any:
    # Create abstract states:
    args = {
        key: ocp.args.PyTreeRestore(value) for key, value in data.items()
    }

    # Load Checkpoint:
    if restore_iteration is None:
        restore_iteration = manager.latest_step()

    restored = manager.restore(
        restore_iteration,
        args=ocp.args.Composite(
            **args,
        ),
    )

    return restored
