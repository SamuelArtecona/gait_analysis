from typing import Any
from absl import app
import os

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import flax.struct
import flax.serialization

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform, System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_forward_velocity: float = 2.0
    # Penalties / Regularization Terms:
    linear_z_velocity: float = -0.8
    lateral_velocity: float = -1.0
    angular_velocity: float = -1.0
    mechanical_power: float = -2e-4
    torque: float = -2e-3
    termination: float = -1.0
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25


def domain_randomize(sys, rng):
    """Randomizes the mjx.Model."""
    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            key, (1,), minval=gain_range[0], maxval=gain_range[1]
        ) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })

    return sys, in_axes


class BarkourEnv(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        velocity_target: float,
        filename: str = 'barkour/scene_mjx.xml',
        config: RewardConfig = RewardConfig(),
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        **kwargs,
    ):
        filename = f'models/{filename}'
        self.filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
            filename,
        )
        sys = mjcf.load(self.filepath)
        self.step_dt = 0.02  # this environment is 50 fps
        sys = sys.tree_replace({'opt.timestep': 0.004})

        # override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.velocity_target = velocity_target

        self.kernel_sigma = config.kernel_sigma
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        self.reward_config = config_dict

        self.torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso'
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.joint_lb = jnp.array([-0.7, -1.0, 0.05] * 4)
        self.joint_ub = jnp.array([0.52, 2.1, 2.1] * 4)
        feet_site = [
            'foot_front_left',
            'foot_hind_left',
            'foot_front_right',
            'foot_hind_right',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self.feet_site_idx = np.array(feet_site_id)
        lower_leg_body = [
            'lower_leg_front_left',
            'lower_leg_hind_left',
            'lower_leg_front_right',
            'lower_leg_hind_right',
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(
            id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
        self.lower_leg_body_idx = np.array(lower_leg_body_id)
        self.foot_radius = 0.0175
        self._nv = sys.nv
        self.history_length = 15
        self.num_observations = 28


    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self.init_q, self.init_qd)

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'command': self.velocity_target,
            'first_contact': jnp.zeros(4, dtype=bool),
            'previous_contact': jnp.zeros(4, dtype=bool),
            'flight_time': jnp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'kick': jnp.array([0.0, 0.0]),
            'step': 0,
        }

        observation_history = jnp.zeros(
            self.history_length * self.num_observations,
        )
        observation = self.get_observation(
            pipeline_state, state_info, observation_history,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {'total_distance': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, kick_noise_key = jax.random.split(state.info['rng'], 2)

        # Distrubance:
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_key, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # Physics step:
        motor_targets = self.default_ctrl + action * self._action_scale
        motor_targets = jnp.clip(motor_targets, self.joint_lb, self.joint_ub)
        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets,
        )
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        observation = self.get_observation(
            pipeline_state,
            state.info,
            state.obs,
        )
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]

        # foot contact data based on z-position
        # pytype: disable=attribute-error
        foot_pos = pipeline_state.site_xpos[self.feet_site_idx]
        foot_contact_z = foot_pos[:, 2] - self.foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['previous_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['previous_contact']
        first_contact = (state.info['flight_time'] > 0) * contact_filt_mm
        state.info['flight_time'] += self.dt

        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self.torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)
        done |= pipeline_state.x.pos[self.torso_idx - 1, 2] < 0.15

        # reward
        rewards = {
            'tracking_forward_velocity': (
                self._reward_tracking_velocity(x, xd)
            ),
            'lateral_velocity': self._reward_lateral_velocity(x, xd),
            'angular_velocity': self._reward_yaw_rate(x, xd),
            'linear_z_velocity': self._reward_vertical_velocity(xd),
            'mechanical_power': self._reward_mechanical_power(
                joint_velocities, pipeline_state.qfrc_actuator[6:],
            ),
            'torque': self._reward_torques(pipeline_state.qfrc_actuator[6:]),
            'termination': jnp.float64(
                self._reward_termination(done, state.info['step'])
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done, state.info['step'])
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_velocities
        state.info['flight_time'] *= ~contact_filt_mm
        state.info['first_contact'] = first_contact
        state.info['previous_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # Proxy Metrics:
        state.metrics['total_distance'] = math.normalize(
            x.pos[self.torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def get_observation(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        observation_history: jax.Array,
    ) -> jax.Array:
        """
            Observation: [
                yaw_rate,
                projected_gravity,
                relative_motor_positions,
                previous_action,
            ]
        """
        inverse_trunk_rotation = math.quat_inv(pipeline_state.x.rot[0])
        body_frame_yaw_rate = math.rotate(
            pipeline_state.xd.ang[0], inverse_trunk_rotation,
        )[2]
        projected_gravity = math.rotate(
            jnp.array([0, 0, -1]), inverse_trunk_rotation,
        )

        q = pipeline_state.q[7:]

        observation = jnp.concatenate([
            jnp.array([body_frame_yaw_rate]),
            projected_gravity,
            q - self.default_pose,
            state_info['previous_action'],
        ])

        # clip, noise
        observation = (
            jnp.clip(observation, -100.0, 100.0)
            + self._obs_noise
            * jax.random.uniform(
                state_info['rng'],
                observation.shape,
                minval=-1,
                maxval=1,
            )
        )
        # stack observations through time
        observation = jnp.roll(
            observation_history, observation.size
        ).at[:observation.size].set(observation)

        return observation

    # ------------ reward functions----------------
    def _reward_tracking_velocity(
        self, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        base_velocity = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        error = jnp.sum(jnp.square(self.velocity_target - base_velocity[0]))
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_lateral_velocity(
        self, x: Transform, xd: Motion
    ) -> jax.Array:
        # Penalize lateral linear velocity
        base_velocity = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        return jnp.square(base_velocity[1])

    def _reward_yaw_rate(
        self, x: Transform, xd: Motion
    ) -> jax.Array:
        # Penalize angular velocity
        base_yaw_rate = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        return jnp.square(base_yaw_rate[2])
    
    def _reward_vertical_velocity(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_mechanical_power(
        self, joint_velocities: jax.Array, torques: jax.Array
    ) -> jax.Array:
        # Mechanical Power Reward:
        return jnp.sum(jnp.maximum(torques * joint_velocities, 0.0))
    
    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))
    
    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)


envs.register_environment('barkour_gait', BarkourEnv)


def main(argv=None):
    env = BarkourEnv(velocity_target=0.25, filename='barkour/scene_hfield_mjx.xml')
    rng = jax.random.PRNGKey(0)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(rng)

    num_steps = 100
    states = []
    for i in range(num_steps):
        print(f"Step: {i}")
        state = step_fn(state, jnp.zeros_like(env.default_ctrl))
        states.append(state.pipeline_state)

    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=states,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
