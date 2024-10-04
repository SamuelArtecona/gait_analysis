from typing import Any, Tuple

import jax
import jax.numpy as jnp

from src.algorithms.ppo import network_utilities as ppo_networks
from src import module_types as types


def calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    truncation_mask: jnp.ndarray,
    termination_mask: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates the Generalized Advantage Estimation."""
    values_ = jnp.concatenate(
        [values, jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    deltas = rewards + gamma * termination_mask * values_[1:] - values_[:-1]
    deltas *= truncation_mask

    initial_gae = jnp.zeros_like(bootstrap_value)

    def scan_loop(carry, xs):
        gae = carry
        truncation_mask, termination_mask, delta = xs
        gae = (
            delta
            + gamma * gae_lambda * termination_mask * truncation_mask * gae
        )
        return gae, gae

    _, vs = jax.lax.scan(
        scan_loop,
        initial_gae,
        (truncation_mask, termination_mask, deltas),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs, values)
    vs_ = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    advantages = (
        rewards
        + gamma * termination_mask * vs_ - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def loss_function(
    params: ppo_networks.PPONetworkParams,
    normalization_params: Any,
    data: types.Transition,
    rng_key: types.PRNGKey,
    ppo_networks: ppo_networks.PPONetworks,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    # Unpack PPO networks:
    action_distribution = ppo_networks.action_distribution
    policy_apply = ppo_networks.policy_network.apply
    value_apply = ppo_networks.value_network.apply

    # Reorder data: (B, T, ...) -> (T, B, ...)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    logits = policy_apply(
        normalization_params, params.policy_params, data.observation,
    )
    values = value_apply(
        normalization_params, params.value_params, data.observation,
    )
    bootstrap_values = value_apply(
        normalization_params, params.value_params, data.next_observation[-1],
    )

    # Be careful with these definitions:
    # Create masks for truncation and termination:
    rewards = data.reward
    truncation_mask = 1 - data.extras['state_data']['truncation']

    # These formulations do not make sense...
    # termination_mask = (1 - data.termination) * truncation_mask
    # Brax formulation:
    # termination_mask = 1 - data.termination * truncation_mask

    # This formulation makes sense: (This performed better in my experiments)
    termination_mask = 1 - data.termination

    # Calculate GAE:
    vs, advantages = calculate_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_values,
        truncation_mask=truncation_mask,
        termination_mask=termination_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    if normalize_advantages:
        advantages = (
            (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        )

    # Calculate ratios:
    log_prob = action_distribution.log_prob(
        logits,
        data.extras['policy_data']['raw_action'],
    )
    previous_log_prob = data.extras['policy_data']['log_prob']
    log_ratios = log_prob - previous_log_prob
    ratios = jnp.exp(log_ratios)

    # Policy Loss:
    unclipped_loss = ratios * advantages
    clipped_loss = advantages * jnp.clip(
        ratios,
        1.0 - clip_coef,
        1.0 + clip_coef,
    )
    policy_loss = -jnp.mean(jnp.minimum(unclipped_loss, clipped_loss))

    # Value Loss:
    value_loss = value_coef * jnp.mean(
        jnp.square(vs - values),
    )

    # Entropy Loss:
    entropy = action_distribution.entropy(
        logits,
        rng_key,
    )
    entropy_loss = -entropy_coef * jnp.mean(
        entropy,
    )

    loss = policy_loss + value_loss + entropy_loss

    return loss, {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
    }
