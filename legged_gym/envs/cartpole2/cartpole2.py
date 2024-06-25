from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from .cartpole2_config import Cartpole2Config


class Cartpole2Task(BaseTask):
    def __init__(self, cfg: Cartpole2Config, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg  # 主要配置文件
        self.sim_params = sim_params  # 仿真参数配置
        self.init_done = False  # 是否初始化完成
        self._parse_cfg(self.cfg)  # 解析配置文件

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        # 初始化父类，和C++不一样，父类的初始化不一定在最开始

        if not self.headless:  # 设置渲染窗口的相机初始化位置
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()  # 初始化pytorch的buffer
        self._prepare_reward_function()  # 准备奖励函数
        self.init_done = True  # 初始化完成

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()
            这个函数是仿真的核心函数，每一步仿真都会调用这个函数.
            (1) 该函数首先接受由策略提供的action，并计算出对应的torque;
            (2) 然后将torque应用到仿真中，进行一次仿真;
            (3) 仿真完成后，调用post_physics_step()函数，检查终止条件，计算奖励，更新观测等。
            (4) 最后返回观测，奖励，终止标志等信息。
            强化学习仿真环境的编写主要过程包括：
            应用action，仿真，计算奖励，检查终止条件，更新观测，返回信息。
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions  # action的范围
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 限制action的范围，防止过大或过小
        # step physics and render each frame
        self.render()  # 渲染
        for _ in range(self.cfg.control.decimation):
            # 计算torque，将其应用在关节上，并进行一次仿真，decimation是控制频率的参数
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)  # 计算torque
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )  # 将torque应用到仿真中
            self.gym.simulate(self.sim)  # 进行一次仿真
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节状态

        self.post_physics_step()  # 检查终止条件，计算奖励，更新观测等

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)  # 对观测进行裁剪，防止过大或过小
        if self.privileged_obs_buf is not None:  # 本实验中没有使用到特权观测信息
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节状态

        self.episode_length_buf += 1  # 记录每个episode的长度
        self.common_step_counter += 1  # 记录总的步数

        # compute observations, rewards, resets, ...
        self.check_termination()  # 检查是否需要终止
        self.compute_reward()  # 计算奖励
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 从reset buffer中找到需要重置的环境的id
        self.reset_idx(env_ids)  # 重置一些环境
        self.compute_observations()  # 计算观测

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.zeros_like(self.episode_length_buf)  # 重置buffer
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf  # 如果超过最大episode长度，需要重置

        cart_pos = self.dof_pos[:, 0]  # 获取小车位置
        pole_angle = self.dof_pos[:, 1]  # 获取杆1角度
        pole_angle2 = self.dof_pos[:, 2]  # 获取杆2角度
        self.reset_buf |= torch.abs(cart_pos) > 3.0  # 如果小车位置超过3，需要重置
        self.reset_buf |= torch.abs(pole_angle) > torch.pi / 2  # 如果杆1角度超过90度，需要重置
        self.reset_buf |= torch.abs(pole_angle2) > torch.pi / 2  # 如果杆2角度超过90度，需要重置

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:  # 如果没有需要重置的环境，直接返回
            return

        # reset robot states
        self._reset_dofs(env_ids)  # 重置关节位置和速度

        # reset buffers
        self.episode_length_buf[env_ids] = 0  # 重置episode长度
        self.reset_buf[env_ids] = 1  # 设置buffer，后续policy需要知道哪些环境被重置了

        # fill extras，记录一些奖励相关的信息，供log使用
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        # 计算奖励/惩罚，使用一些神奇的map操作就将文件结尾处的一堆奖励函数给调用了，
        # 涉及到了一些python的神奇语法，看不懂就算了，
        # 知道是依次调用结尾的一堆奖励函数即可。
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # 同样是计算奖励，这里计算的是智能体被重置时才获取的额外奖励/惩罚
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置
                self.dof_vel * self.obs_scales.dof_vel,  # 关节速度
            ),
            dim=-1,
        )

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )  # 创建仿真器handle
        self._create_ground_plane()  # 创建地面
        self._create_envs()  # 创建智能体

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)  # 设置渲染器初始位姿

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 根据policy输出计算力矩，这里仅简单按比例直出了，训练机器人时可以加一些控制器如pd等
        actions_scaled = actions * self.cfg.control.action_scale
        torques = torch.zeros_like(self.torques)
        torques[:, 0:1] = actions_scaled

        return torques

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]  # 复位时随机指定位置和速度

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )  # 将新关节位置应用到仿真中

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 从仿真器获取关节状态的矩阵

        self.gym.refresh_dof_state_tensor(self.sim)  # 刷新关节状态

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 将关节状态矩阵包装成pytorch tensor
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 将关节状态矩阵分成位置和速度两部分

        # the following tensors are NOT used in this experiment, but are kept for compatibility with other tasks
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_forces = torch.zeros(
            self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_indices = torch.tensor([0, 1], dtype=torch.int32, device=self.device, requires_grad=False)

        # initialize some data used later on
        self.common_step_counter = 0  # 记录总的步数
        self.extras = {}  # 用于记录一些额外信息

        self.torques = torch.zeros(  # 力矩(直接应用于关节)
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.actions = torch.zeros(  # 动作(网络模型输出)
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # 关节默认位置为0
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 用一堆神奇的map操作将奖励函数给包装进一个map里了，
        # 涉及到了一些python的神奇语法，看不懂就算了，
        # 知道是依次调用结尾的一堆奖励函数即可。
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 地面法向量为z轴
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        # 设置地面的参数(本实验没有使用到地面的信息，这里只是出于兼容性和完整性考虑保留了)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)  # 设置URDF文件名和文件路径

        asset_options = gymapi.AssetOptions()  # 设置机器人模型的各种属性
        # 属性的具体含义可以参考isaacgym的文档
        # https://blog.zzshub.cn/legged_gym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 加载模型
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # 获取关节数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # 获取刚体数量
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)  # 获取关节名称

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)  # 设置机器人初始位置

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(self.cfg.env.env_spacing, self.cfg.env.env_spacing, 0.0)  # 设置环境边界
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            # 创建环境实例, Isaac Gym是一个并行化的仿真器，所以这里会创建多个环境实例
            # 每个环境有一个env_handler, 每个机器人有一个actor_handle, 每个actor_handle对应一个机器人实例
            # 由于这里每个环境只有一个机器人，所以只有一个actor_handle,
            # 需要多智能体组合时，可以在一个env_handle中创建多个actor_handle
            # 不同环境的机器人无法进行交互(会穿模而不是撞在一起); 同一个环境的机器人可以进行交互
            # 前面的asset是机器人模型，首先导入这个模型，并修改模型的相关参数，之后创建机器人实例

            # 创建环境
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 在环境中创建机器人实例
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )

            # 设置关节驱动模式
            dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)  # 获取关节属性
            dof_props["driveMode"][0] = gymapi.DOF_MODE_EFFORT  # 设置1关节(小车关节)驱动模式为力矩驱动
            dof_props["driveMode"][1] = gymapi.DOF_MODE_NONE  # 设置2关节(杆1关节)驱动模式为无驱动
            dof_props["driveMode"][2] = gymapi.DOF_MODE_NONE  # 设置3关节(杆2关节)驱动模式为无驱动
            dof_props["stiffness"][:] = 0.0
            dof_props["damping"][:] = 0.0  # 设置关节pd为0
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)  # 设置关节属性

            self.envs.append(env_handle)  # 记录环境handle
            self.actor_handles.append(actor_handle)  # 记录机器人实例handle

    def _parse_cfg(self, cfg):
        # 设置仿真时间步长等参数
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    # ------------ reward functions----------------

    def _reward_joint_angle(self):
        # 设置关节角度奖励，这里是一个简单的奖励函数，
        # 奖励为关节角度的平方，这里的奖励为正，但是乘上scales后为负，
        # 关节角度越大，奖励越小
        pole_angle = self.dof_pos[:, 1]
        pole_angle2 = self.dof_pos[:, 2]
        reward = pole_angle * pole_angle + pole_angle2 * pole_angle2 * 2
        return reward

    def _reward_joint_velocity(self):
        # 设置关节速度奖励，这里是一个简单的奖励函数，
        # 奖励为关节速度的平方，这里的奖励为正，但是乘上scales后为负，
        # 关节速度越大，奖励越小，限制关节速度的大小
        cart_vel = self.dof_vel[:, 0]
        pole_vel = self.dof_vel[:, 1]
        pole_vel2 = self.dof_vel[:, 2]

        reward = torch.abs(cart_vel) + torch.abs(pole_vel) + torch.abs(pole_vel2)
        return reward

    def _reward_termination(self):
        # 设置终止奖励，当倒立摆倒下的时候需要给一点惩罚
        cart_pos = self.dof_pos[:, 0]
        pole_angle = self.dof_pos[:, 1]
        pole_angle2 = self.dof_pos[:, 2]

        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward = torch.where(torch.abs(cart_pos) > 3, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > torch.pi / 2, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle2) > torch.pi / 2, torch.ones_like(reward) * -2.0, reward)

        return reward
