from typing import Optional, Tuple, Any
import dataclasses
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import flax.linen as nn
import optax
from brax.training.acme import running_statistics, specs
from brax.envs.base import Env

from src.algorithms.ppo import checkpoint_utilities
from src.algorithms.ppo import network_utilities as ppo_networks
from src.algorithms.ppo.network_utilities import PPONetworkParams
from src.algorithms.ppo.checkpoint_utilities import (
    RestoredCheckpoint, TrainState,
)


@dataclasses.dataclass
class Metadata:
    network_metadata: checkpoint_utilities.network_metadata
    loss_metadata: checkpoint_utilities.loss_metadata
    training_metadata: checkpoint_utilities.training_metadata


def load_policy(checkpoint_name: str, environment: Env, restore_iteration: Optional[int] = None):
    # Load Metadata:
    checkpoint_direrctory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        f"checkpoints/{checkpoint_name}",
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'network_metadata',
            'loss_metadata',
            'training_metadata',
        ),
    )

    metadata = checkpoint_utilities.load_checkpoint(
        manager=manager,
        restore_iteration=restore_iteration,
        network_metadata=checkpoint_utilities.empty_network_metadata(),
        loss_metadata=checkpoint_utilities.empty_loss_metadata(),
        training_metadata=checkpoint_utilities.empty_training_metadata(),
    )
    network_metadata = metadata.network_metadata
    loss_metadata = metadata.loss_metadata
    training_metadata = metadata.training_metadata

    env = environment

    # Restore Networks:
    policy_layer_sizes = (network_metadata.policy_layer_size,) * network_metadata.policy_depth
    value_layer_sizes = (network_metadata.value_layer_size,) * network_metadata.value_depth
    if training_metadata.normalize_observations:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=policy_layer_sizes,
        value_layer_sizes=value_layer_sizes,
        activation=eval(network_metadata.activation),
        kernel_init=eval(network_metadata.kernel_init),
    )
    optimizer = eval(training_metadata.optimizer)

    # Create Keys and Structures:
    key = jax.random.key(training_metadata.seed)
    init_params = PPONetworkParams(
        policy_params=network.policy_network.init(key),
        value_params=network.value_network.init(key),
    )

    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            specs.Array(env.observation_size, jnp.dtype('float32'))
        ),
        env_steps=0,
    )

    # Restore Train State:
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'train_state',
        ),
    )
    restored_train_state = checkpoint_utilities.load_checkpoint(
        manager=manager,
        restore_iteration=restore_iteration,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    # Construct Policy:
    make_policy = ppo_networks.make_inference_fn(ppo_networks=network)
    params = (
        train_state.normalization_params, train_state.params.policy_params,
    )

    return make_policy, params


def load_checkpoint(
    checkpoint_name: str,
    environment: Env,
    restore_iteration: Optional[int] = None,
) -> Tuple[RestoredCheckpoint, Metadata]:
    # Load Metadata:
    checkpoint_direrctory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        f"checkpoints/{checkpoint_name}",
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'network_metadata',
            'loss_metadata',
            'training_metadata',
        ),
    )

    metadata = checkpoint_utilities.load_checkpoint(
        manager=manager,
        restore_iteration=restore_iteration,
        network_metadata=checkpoint_utilities.empty_network_metadata(),
        loss_metadata=checkpoint_utilities.empty_loss_metadata(),
        training_metadata=checkpoint_utilities.empty_training_metadata(),
    )
    network_metadata = metadata.network_metadata
    loss_metadata = metadata.loss_metadata
    training_metadata = metadata.training_metadata

    env = environment

    # Restore Networks:
    policy_layer_sizes = (network_metadata.policy_layer_size,) * network_metadata.policy_depth
    value_layer_sizes = (network_metadata.value_layer_size,) * network_metadata.value_depth
    if training_metadata.normalize_observations:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=policy_layer_sizes,
        value_layer_sizes=value_layer_sizes,
        activation=eval(network_metadata.activation),
        kernel_init=eval(network_metadata.kernel_init),
    )
    optimizer = eval(training_metadata.optimizer)

    # Create Keys and Structures:
    key = jax.random.key(training_metadata.seed)
    init_params = PPONetworkParams(
        policy_params=network.policy_network.init(key),
        value_params=network.value_network.init(key),
    )

    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            specs.Array(env.observation_size, jnp.dtype('float32'))
        ),
        env_steps=0,
    )

    # Restore Train State:
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'train_state',
        ),
    )
    restored_train_state = checkpoint_utilities.load_checkpoint(
        manager=manager,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    # Kind of redundant, but this gives us type hint instead of Any
    metadata = Metadata(
        network_metadata=network_metadata,
        loss_metadata=loss_metadata,
        training_metadata=training_metadata,
    )

    return (
        RestoredCheckpoint(network=network, train_state=train_state), metadata
    )
