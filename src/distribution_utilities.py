from typing import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import distrax

import src.module_types as types


@dataclass
class ParametricDistribution():
    """
        Wrapper Class around Distrax Distributions.

        Args:
            distribution: Callable[..., distrax.Distribution]
            bijector: distrax.Bijector = distrax.Lambda(lambda x: x)
            min_std: float = 1e-3
            var_scale: float = 1.0
    """
    distribution: Callable[..., distrax.Distribution]
    bijector: distrax.Bijector = distrax.Lambda(lambda x: x)
    min_std: float = 1e-3
    var_scale: float = 1.0

    def create_distribution(self, params: jnp.ndarray) -> distrax.Distribution:
        loc, scale = jnp.split(params, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self.min_std) * self.var_scale
        return distrax.Transformed(
            distribution=self.distribution(loc=loc, scale=scale),
            bijector=self.bijector,
        )

    def entropy(
        self,
        params: jax.Array,
        rng_key: types.PRNGKey,
    ) -> jnp.ndarray:
        transformed_distribution = self.create_distribution(params=params)
        sample = transformed_distribution.distribution.sample(seed=rng_key)
        entropy = transformed_distribution.distribution.entropy()
        forward_log_det_jacobian = self.bijector.forward_log_det_jacobian(
            sample,
        )
        entropy = entropy + forward_log_det_jacobian
        return jnp.sum(entropy, axis=-1)

    def log_prob(
        self,
        params: jax.Array,
        sample: jax.Array,
    ) -> jnp.ndarray:
        transformed_distribution = self.create_distribution(params=params)
        log_probs = transformed_distribution.distribution.log_prob(sample)
        log_probs -= self.bijector.forward_log_det_jacobian(sample)
        return jnp.sum(log_probs, axis=-1)

    def mode(
        self,
        params: jax.Array,
    ) -> jnp.ndarray:
        transformed_distribution = self.create_distribution(params=params)
        return self.bijector.forward(transformed_distribution.distribution.mode())

    def base_distribution_sample(
        self,
        params: jax.Array,
        rng_key: types.PRNGKey,
    ) -> jnp.ndarray:
        transformed_distribution = self.create_distribution(params=params)
        return transformed_distribution.distribution.sample(seed=rng_key)

    def process_sample(
        self,
        sample: jax.Array,
    ) -> jnp.ndarray:
        return self.bijector.forward(sample)
