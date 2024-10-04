import dataclasses

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

# Custom types:
import src.module_types as types
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_normalization: bool = False

    @nn.compact
    def __call__(self, x: jax.Array):
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                features=layer_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                if self.layer_normalization:
                    x = nn.LayerNorm(name=f"layer_norm_{i}")(x)
        return x


def make_policy_network(
    input_size: int,
    output_size: int,
    input_normalization_fn: types.InputNormalizationFn = types
    .identity_normalization_fn,
    layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.tanh,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    bias: bool = True,
    layer_normalization: bool = False,
) -> FeedForwardNetwork:
    """Intializes a policy network."""
    policy_network = MLP(
        layer_sizes=list(layer_sizes) + [output_size],
        activation=activation,
        kernel_init=kernel_init,
        bias=bias,
        layer_normalization=layer_normalization,
    )

    def apply(normalization_params, policy_params, x):
        x = input_normalization_fn(x, normalization_params)
        return policy_network.apply(policy_params, x)

    dummy_input = jnp.zeros((1, input_size))
    return FeedForwardNetwork(
        init=lambda key: policy_network.init(key, dummy_input), apply=apply,
    )


def make_value_network(
    input_size: int,
    input_normalization_fn: types.InputNormalizationFn = types
    .identity_normalization_fn,
    layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.tanh,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    bias: bool = True,
    layer_normalization: bool = False,
) -> FeedForwardNetwork:
    """Intializes a value network."""
    value_network = MLP(
        layer_sizes=list(layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
        bias=bias,
        layer_normalization=layer_normalization,
    )

    def apply(normalization_params, value_params, x):
        x = input_normalization_fn(x, normalization_params)
        return jnp.squeeze(value_network.apply(value_params, x), axis=-1)

    dummy_input = jnp.zeros((1, input_size))
    return FeedForwardNetwork(
        init=lambda key: value_network.init(key, dummy_input), apply=apply,
    )
