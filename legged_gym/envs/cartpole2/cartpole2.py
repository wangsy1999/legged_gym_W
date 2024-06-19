import sys
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import numpy as np
import torch
from legged_gym.envs.base.base_task import BaseTask
from .cartpole2_config import Cartpole2Config


class Cartpole2Task(BaseTask):
    def __init__(self, cfg: Cartpole2Config, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.dt = self.sim_params.dt * self.cfg.control.decimation
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self.init_buffers()

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # define plane on which environments are initialized
        spacing = self.cfg.env.env_spacing
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../resources/robots")
        asset_file = "cartpole/cartpole2.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.cfg.assets.fix_base_link

        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)
        self.dof_names = self.gym.get_asset_dof_names(cartpole_asset)

        pose = gymapi.Transform()
        pose.p.z = 2.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, 25)
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props["driveMode"][0] = gymapi.DOF_MODE_EFFORT
            dof_props["driveMode"][1] = gymapi.DOF_MODE_NONE
            dof_props["driveMode"][2] = gymapi.DOF_MODE_NONE
            dof_props["stiffness"][:] = 0.0
            dof_props["damping"][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def init_buffers(self):
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.extras = {}
        self.obs_clip = self.cfg.assets.clip_observations
        self.act_clip = self.cfg.assets.clip_actions

    def step(self, actions):
        clip_actions = torch.clip(actions, -self.act_clip, self.act_clip).squeeze()
        action_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        action_tensor[:: self.num_dof] = clip_actions.to(self.device).squeeze() * self.cfg.control.action_scale
        forces = gymtorch.unwrap_tensor(action_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        if not self.headless:
            self.render()

        for _ in range(self.cfg.control.decimation):
            self.gym.simulate(self.sim)
        self.episode_length_buf += 1

        self.compute_observations()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        clip_obs = self.cfg.assets.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = None

        return (self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras)

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = self.time_out_buf

        cart_pos = self.dof_pos[:, 0]
        pole_angle = self.dof_pos[:, 1]
        pole_angle2 = self.dof_pos[:, 2]
        reset_dist = self.cfg.control.resetDist

        self.reset_buf = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            torch.abs(pole_angle2) > np.pi / 2, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.obs_buf[:, 0] = self.dof_pos[:, 0].squeeze()
        self.obs_buf[:, 1] = self.dof_vel[:, 0].squeeze()
        self.obs_buf[:, 2] = self.dof_pos[:, 1].squeeze()
        self.obs_buf[:, 3] = self.dof_vel[:, 1].squeeze()
        self.obs_buf[:, 4] = self.dof_pos[:, 2].squeeze()
        self.obs_buf[:, 5] = self.dof_vel[:, 2].squeeze()

    def compute_reward(self):
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]
        pole_angle2 = self.obs_buf[:, 4]
        pole_vel2 = self.obs_buf[:, 5]
        reset_dist = self.cfg.control.resetDist

        reward = (
            2.0
            - pole_angle * pole_angle
            - pole_angle2 * pole_angle2
            - 0.01 * torch.abs(cart_vel)
            - 0.005 * torch.abs(pole_vel)
            - 0.01 * torch.abs(cart_vel)
            - 0.005 * torch.abs(pole_vel2)
        )

        # adjust reward for reset agents
        reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle2) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        self.rew_buf = reward

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
