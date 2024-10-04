from typing import Any, Callable, Sequence, Tuple, NamedTuple, Protocol, Mapping, TypeVar

import jax.numpy as jnp
from brax import envs

Params = Any
PRNGKey = jnp.ndarray
NomralizationParams = Any
NetworkParams = Tuple[NomralizationParams, Params]
PolicyParams = Tuple[NomralizationParams, Params]
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

Observation = jnp.ndarray
Action = jnp.ndarray
PolicyData = Mapping[str, Any]
Metrics = Mapping[str, Any]

State = envs.State
Env = envs.Env

NetworkType = TypeVar('NetworkType')


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    termination: jnp.ndarray
    next_observation: jnp.ndarray
    extras: Mapping[str, Any]


class Policy(Protocol):
    def __call__(
        self,
        x: jnp.ndarray,
        key: PRNGKey,
    ) -> Tuple[Action, PolicyData]:
        pass


class InputNormalizationFn(Protocol):
    def __call__(
        self,
        x: jnp.ndarray,
        normalization_params: NomralizationParams
    ) -> jnp.ndarray:
        pass


def identity_normalization_fn(
    x: jnp.ndarray,
    normalization_params: NomralizationParams
) -> jnp.ndarray:
    del normalization_params
    return x


class NetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        input_normalization_fn: InputNormalizationFn = identity_normalization_fn,
    ) -> NetworkType:
        pass
