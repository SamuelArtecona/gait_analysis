from typing import Callable, List, Optional, Union
import os
from pathlib import Path
import time

import jax
import numpy as np

import cv2 as cv
import mujoco

import brax
from brax import envs, base
from brax.io import image

import src.module_types as types
from src.training_utilities import unroll_policy_steps, render_policy


class Evaluator:

    def __init__(
        self,
        env: envs.Env,
        policy_generator: Callable[[types.PolicyParams], types.Policy],
        num_envs: int,
        episode_length: int,
        action_repeat: int,
        key: types.PRNGKey,
    ):
        self.key = key
        self.walltime = 0.0

        env = envs.training.EvalWrapper(env)

        def _evaluation_loop(
            policy_params: types.PolicyParams,
            key: types.PRNGKey,
        ) -> types.State:
            reset_keys = jax.random.split(key, num_envs)
            initial_state = env.reset(reset_keys)
            final_state, _ = unroll_policy_steps(
                env,
                initial_state,
                policy_generator(policy_params),
                key,
                num_steps=episode_length // action_repeat,
            )
            return final_state

        self.evaluation_loop = jax.jit(_evaluation_loop)
        self.steps_per_epoch = episode_length * num_envs

    def evaluate(
        self,
        policy_params: types.PolicyParams,
        training_metrics: types.Metrics,
        aggregate_episodes: bool = True,
    ) -> types.Metrics:
        self.key, subkey = jax.random.split(self.key)

        start_time = time.time()
        state = self.evaluation_loop(policy_params, subkey)
        evaluation_metrics = state.info['eval_metrics']
        evaluation_metrics.active_episodes.block_until_ready()
        epoch_time = time.time() - start_time
        metrics = {}
        for func in [np.mean, np.std]:
            suffix = '_std' if func == np.std else ''
            metrics.update({
                f'eval/episode_{name}{suffix}': (
                    func(value) if aggregate_episodes else value
                )
                for name, value in evaluation_metrics.episode_metrics.items()
            })

        metrics['eval/avg_episode_length'] = np.mean(
            evaluation_metrics.episode_steps,
        )
        metrics['eval/epoch_time'] = epoch_time
        metrics['eval/steps_per_second'] = self.steps_per_epoch / epoch_time
        self.walltime = self.walltime + epoch_time
        metrics = {
            'eval/walltime': self.walltime,
            **training_metrics,
            **metrics,
        }

        return metrics


class Renderer:

    def __init__(
        self,
        env: envs.Env,
        policy_generator: Callable[[types.PolicyParams], types.Policy],
        episode_length: int,
        action_repeat: int,
        filepath: str,
        key: types.PRNGKey,
    ):
        # Make Directory to store videos:
        self.filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
            f"visualization/{filepath}",
        )
        Path(self.filepath).mkdir(parents=True, exist_ok=True)

        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0]
        camera.azimuth = 60
        camera.elevation = -20
        camera.distance = 5
        self.camera = camera

        self.key = key
        self.walltime = 0.0

        env = envs.training.EvalWrapper(env)
        self.dt = env.dt
        self.sys = env.sys

        def _render_loop(
            policy_params: types.PolicyParams,
            key: types.PRNGKey,
        ) -> types.State:
            reset_keys = jax.random.split(key)
            initial_state = env.reset(reset_keys)
            states = render_policy(
                env,
                initial_state,
                policy_generator(policy_params),
                key,
                num_steps=episode_length // action_repeat,
            )

            return states

        self.render_loop = jax.jit(_render_loop)
        self.steps_per_epoch = episode_length

    def render(
        self,
        policy_params: types.PolicyParams,
        iteration: int,
    ) -> None:
        def create_video(
            trajectory: Union[List[base.State], base.State],
            filepath: str,
            iteration: int,
            height: int = 240,
            width: int = 320,
            camera: Optional[str] = None,
        ) -> None:
            # Setup Animation Writer:
            FPS = int(1 / self.dt)

            # Create and set context for mujoco rendering:
            ctx = mujoco.GLContext(width, height)
            ctx.make_current()

            # Create Dummy Data and Generate Frames:
            state_list = []
            data = mujoco.mjx.make_data(self.sys.mj_model)
            num_steps = trajectory.q.shape[0]
            for i in range(num_steps):
                state_list.append(
                    brax.mjx.base.State(
                        q=trajectory.q[i],
                        qd=trajectory.qd[i],
                        x=brax.base.Transform(
                            pos=trajectory.x.pos[i],
                            rot=trajectory.x.rot[i],
                        ),
                        xd=brax.base.Motion(
                            vel=trajectory.xd.vel[i],
                            ang=trajectory.xd.ang[i],
                        ),
                        **data.__dict__,
                    ),
                )
            frames = image.render_array(
                sys=self.sys,
                trajectory=state_list,
                height=height,
                width=width,
                camera=camera,
            )

            filename = os.path.join(filepath, f'{iteration}.mp4')
            out = cv.VideoWriter(
                filename=filename,
                fourcc=cv.VideoWriter_fourcc(*'mp4v'),
                fps=FPS,
                frameSize=(width, height),
                isColor=True,
            )

            num_frames = np.shape(frames)[0]
            for i in range(num_frames):
                out.write(frames[i])

            out.release()

        self.key, subkey = jax.random.split(self.key)

        state = self.render_loop(policy_params, subkey)

        create_video(
            trajectory=state,
            filepath=self.filepath,
            iteration=iteration,
            camera=self.camera,
        )
