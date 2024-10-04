from typing import Any, List, Sequence
import os

import jax
import jax.numpy as jnp
import numpy as np

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from ml_collections import config_dict
import mujoco


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


def get_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1.5,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.8,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-2.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.0002,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.2,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.1,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config


class BarkourEnv(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'barkour/scene_mjx.xml',
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

        self.reward_config = get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso'
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        self._default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.lowers = jnp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jnp.array([0.52, 2.1, 2.1] * 4)
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
        self._feet_site_id = np.array(feet_site_id)
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
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            'rng': rng,
            'last_act': jnp.zeros(12),
            'last_vel': jnp.zeros(12),
            'command': self.sample_command(key),
            'last_contact': jnp.zeros(4, dtype=bool),
            'feet_air_time': jnp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            'kick': jnp.array([0.0, 0.0]),
            'step': 0,
        }

        obs_history = jnp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jnp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        state = State(pipeline_state, obs, reward, done, metrics,
                      state_info)  # pytype: disable=wrong-arg-types
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        # pytype: disable=attribute-error
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt

        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.lowers)
        done |= jnp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            # pytype: disable=attribute-error
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # If contact triggered:
        # impact_window = 5
        # impact_counter = first_contact * impact_counter
        # impact_filter = first_contact * (impact_counter < impact_window)
        # impact_counter += 1
        # impact_mask = (1 - impact_filter)
        # filtered_rewards = {
        #     k: v * self.reward_config.rewards.scales[k] for k, v in filtered_rewards.items()
        # }
        # filtered_reward = jnp.clip(sum(filtered_rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jnp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(
            x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        # done = jnp.float32(done)
        done = jnp.float64(done)

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jnp.concatenate([
            jnp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
            # projected gravity
            math.rotate(jnp.array([0, 0, -1]), inv_torso_rot),
            state_info['command'] * jnp.array([2.0, 2.0, 0.25]),  # command
            pipeline_state.q[7:] - self._default_pose,           # motor angles
            state_info['last_act'],                              # last action
        ])

        # clip, noise
        obs = jnp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info['rng'], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jnp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, act: jax.Array, last_act: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jnp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def render(
        self, trajectory: List[base.State], camera: str | None = None
    ) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera)

    def get_observation(
        self,
        mj_data: mujoco.MjData,
        command: np.ndarray,
        previous_action: np.ndarray,
        observation_history: np.ndarray,
    ) -> np.ndarray:
        # Numpy implementation of the observation function:
        def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
            if len(vec.shape) != 1:
                raise ValueError('vec must have no batch dimensions.')
            s, u = quat[0], quat[1:]
            r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
            r = r + 2 * s * np.cross(u, vec)
            return r

        def quat_inv(q: np.ndarray) -> np.ndarray:
            return q * np.array([1, -1, -1, -1])

        base_w = mj_data.qpos[3:7]
        base_dw = mj_data.qvel[3:6]
        joint_q = mj_data.qpos[7:]

        inverse_torso_rotation = quat_inv(base_w)
        local_dw = rotate(base_dw, inverse_torso_rotation)

        new_observation = np.concatenate([
            np.array([local_dw[2]]) * 0.25,
            rotate(np.array([0, 0, -1]), inverse_torso_rotation),
            command * np.array([2.0, 2.0, 0.25]),
            joint_q - self._default_pose,
            previous_action,
        ])

        # clip, noise
        new_observation = (
            jnp.clip(new_observation, -100.0, 100.0)
            + self._obs_noise * np.random.uniform(
                low=-1, high=1, size=new_observation.shape,
            )
        )
        # stack observations through time
        observation = np.roll(observation_history, new_observation.size)
        observation[:new_observation.size] = new_observation

        return observation


envs.register_environment('barkour', BarkourEnv)