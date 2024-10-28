# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# Copyright (c) 2024 ZhouZishun

from legged_gym.envs.base.base_config import BaseConfig


class Pf2Cfg(BaseConfig):
    """
    Configuration class for the pf2 humanoid robot.
    """

    class env:
        # change the observation dim
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        frame_stack = 50
        c_frame_stack = 6
        num_single_obs = 29
        num_observations = int(frame_stack * num_single_obs + 3)
        single_num_privileged_obs = 49
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 6
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset:
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pf2/urdf/pf2.urdf"
        name = "pf2"
        foot_name = "foot"
        knee_name = "tib"

        terminate_after_contacts_on = ["torso", "fem", "tib"]
        penalize_contacts_on = ["tib"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
        collapse_fixed_joints = False

    class terrain:
        measured_points_x = [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # trimesh only:
        # mesh_type = "trimesh"
        mesh_type = "plane"
        curriculum = True
        # rough terrain only:
        measure_heights = False
        static_friction = 1
        dynamic_friction = 1
        terrain_length = 10.0
        terrain_width = 10.0
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)
        max_init_terrain_level = 2  # starting curriculum state
        # up slope, down slope, uniform noisy ground, up stair, down stair, sine wave, obstacle, flat ground
        # terrain_proportions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        terrain_proportions = [0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0]
        restitution = 0.0
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.01  # [m]
        slope_treshold = 0.5  # 30°
        border_size = 10

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.3
            lin_vel = 0.0
            # quat = 0.03
            quat = 0.1
            height_measurements = 0.1

    class init_state:
        pos = [0.0, 0.0, 0.73]
        rot = [0.0, 0, 0.0, 1]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "lhiproll": 0.0,
            "lfempitch": 0.3,
            "ltibpitch": -0.6,
            # "lwheelrot": 0.0,
            "rhiproll": -0.0,
            "rfempitch": 0.3,
            "rtibpitch": -0.6,
            # "rwheelrot": 0.0,
        }

    class control:
        # PD Drive parameters:
        stiffness = {"hiproll": 70.97, "fempitch": 70.3, "tibpitch": 70.7}  # T
        damping = {
            "hiproll": 3.799,
            "fempitch": 3.114,
            "tibpitch": 3.565,
            # "wheelrot": 0.0,
        }  # [N*m/rad] D

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim:
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3.0, 3.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.5
        dynamic_randomization = 0.00
        randomize_base_com = True
        com_x_range = [0.02, 0.07]
        com_y_range = [-0.05, 0.05]
        com_z_range = [-0.00, 0.00]
        randomize_pd = True
        randomize_pd_range = [0.7, 1.2]

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.0
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 0.9]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            # ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

        # class ranges: # for play convienience
        #     x = 1.0
        #     y = 0
        #     yaw = 0
        #     lin_vel_x = [x, x]  # min max [m/s]
        #     lin_vel_y = [y, y]  # min max [m/s]
        #     ang_vel_yaw = [yaw, yaw]  # min max [rad/s]
        #     heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.68
        min_dist = 0.15
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.3  # rad
        target_feet_height = 0.05  # m
        cycle_time = 0.5  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 450  # Forces above this value are penalized

        ###################first stage trainning 3000 epoches###################
        class scales:
            # reference motion tracking
            joint_pos = 3.0
            feet_clearance = 1.0
            feet_contact_number = 1.2
            # gaita
            feet_air_time = 1.0
            foot_slip = -0.05
            feet_distance = 0.2
            # knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.0
            base_height = 0.2
            base_acc = 0.3
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.0

        # ################## second stage trainning 16000 epoches ##############
        # # stage two list
        # # Oct10_09-39-50 -> Oct10_13-33-56
        # # Oct12_18-49-21 -> Oct15_21-56-55
        # # Oct16_22-19-13 -> Oct17_12-42-14
        # # Oct18_16-37-02 -> Oct18_18-51-34
        # class scales:
        #     # reference motion tracking
        #     joint_pos = 2.6
        #     roll_pos = 2.0
        #     feet_clearance = 1.0
        #     feet_contact_number = 1.2
        #     # gait
        #     feet_air_time = 1.0
        #     foot_slip = -0.05
        #     feet_distance = 0.2
        #     # contact
        #     feet_contact_forces = -1.0
        #     # vel tracking
        #     tracking_lin_vel = 1.2
        #     tracking_ang_vel = 1.1
        #     vel_mismatch_exp = 0.5  # lin_z; ang x,y
        #     low_speed = 0.2
        #     track_vel_hard = 0.5
        #     # base pos
        #     default_joint_pos = 0.5
        #     orientation = 1.0
        #     base_height = 0.2
        #     base_acc = 1.2
        #     # energy
        #     action_smoothness = -0.5
        #     torques = -5e-6
        #     dof_vel = -5e-4
        #     dof_acc = -10e-7
        #     collision = -1.0
        #     ang_vel_xy = -0.3
        #     joint_limit = -0.1

        # ################## third stage trainning 3000 epoches ##############
        # class scales:
        #     # reference motion tracking
        #     joint_pos = 1.6
        #     roll_pos = 1.0
        #     feet_clearance = 1.0
        #     feet_contact_number = 1.2
        #     # gait
        #     feet_air_time = 1.0
        #     foot_slip = -0.05
        #     feet_distance = 0.2
        #     # contact
        #     feet_contact_forces = -1.0
        #     # vel tracking
        #     tracking_lin_vel = 4.0  # 2.0
        #     tracking_ang_vel = 3.0  # 2.5
        #     vel_mismatch_exp = 0.5  # lin_z; ang x,y
        #     low_speed = 0.2
        #     track_vel_hard = 1.5
        #     # base pos
        #     default_joint_pos = 0.5
        #     orientation = 1.0
        #     base_height = 0.2
        #     base_acc = 1.2
        #     # energy
        #     action_smoothness = -0.5
        #     torques = -1e-5
        #     dof_vel = -5e-4
        #     dof_acc = -10e-7
        #     collision = -1.0
        #     ang_vel_xy = -0.3
        #     joint_limit = -0.1
        #     no_fly = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0
        action_lower_pos_limit = [-0.20, 0.03, -0.9, -0.20, 0.03, -0.9]
        action_higher_pos_limit = [0.20, 0.9, -0.05, 0.20, 0.9, -0.05]


class Pf2CfgPPO(BaseConfig):
    seed = 98698
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm:
        # training params
        # no need to change this unless you know what you are doing
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01
        max_grad_norm = 1.0

        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 4500  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = "pf2"
        run_name = ""
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
