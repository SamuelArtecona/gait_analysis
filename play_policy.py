from absl import app, flags
import os

import jax
import jax.numpy as jnp

from brax.io import html

from src.envs import unitree_go2_energy as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


def main(argv=None):
    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(velocity_target=0.375)
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

    velocity = 1.0
    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)
    command = [velocity, 0.0, 0.0]
    state.info['command'] = jnp.array(command)

    num_steps = 1000
    states = []
    for i in range(num_steps):
        # Stop Command Sampling:
        state.info['command'] = jnp.array(command)
        key, subkey = jax.random.split(key)
        action, _ = inference_fn(state.obs, subkey)
        state = step_fn(state, action)
        states.append(state.pipeline_state)

    # Generate HTML:
    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=states,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
