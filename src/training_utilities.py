from typing import Sequence, Tuple

import jax

from src.module_types import Transition
import src.module_types as types
from src.module_types import Env, State


def policy_step(
    env: Env,
    state: State,
    policy: types.Policy,
    key: types.PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    actions, policy_data = policy(state.obs, key)
    next_state = env.step(state, actions)
    state_data = {x: next_state.info[x] for x in extra_fields}
    return next_state, Transition(
        observation=state.obs,
        action=actions,
        reward=next_state.reward,
        termination=next_state.done,
        next_observation=next_state.obs,
        extras={
            'policy_data': policy_data,
            'state_data': state_data,
        },
    )


def unroll_policy_steps(
    env: Env,
    state: State,
    policy: types.Policy,
    key: types.PRNGKey,
    num_steps: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    @jax.jit
    def f(carry, unused_t):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, transition = policy_step(
            env,
            state,
            policy,
            key,
            extra_fields=extra_fields,
        )
        return (state, subkey), transition

    (final_state, _), transitions = jax.lax.scan(
        f,
        (state, key),
        (),
        length=num_steps,
    )
    return final_state, transitions


def render_policy(
    env: Env,
    state: State,
    policy: types.Policy,
    key: types.PRNGKey,
    num_steps: int,
    extra_fields: Sequence[str] = (),
) -> State:
    @jax.jit
    def f(carry, unused_t):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, _ = policy_step(
            env,
            state,
            policy,
            key,
            extra_fields=extra_fields,
        )
        pipeline_state = jax.tree.map(lambda x: x[0], state.pipeline_state)
        return (state, subkey), pipeline_state

    _, states = jax.lax.scan(
        f,
        (state, key),
        (),
        length=num_steps,
    )

    return states
