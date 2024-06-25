from legged_gym.envs.base.base_config import BaseConfig


class Cartpole2Config(BaseConfig):
    class env:
        num_envs = 512  # 设置环境数量为512
        num_observations = 6  # 观测值数量为6（三关节速度+位置）
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 1  # 动作数量为1 小车力矩
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class control:
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 300.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class terrain:
        # 地面的摩擦系数等，本实验没用
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class noise:
        # 设置不加观测噪声，本实验每加噪声，也没写加噪声的函数，所以这个参数没用，出于兼容性考虑，保留
        add_noise = False

    class domain_rand:
        # 设置不随机化摩擦系数，本实验没用，也没写随机化摩擦系数的函数，所以这个参数没用，出于兼容性考虑，保留
        randomize_friction = False
        push_robots = False

    class asset:
        # asset的参数设置，具体作用请参阅isaacgym的文档
        # https://blog.zzshub.cn/legged_gym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/cartpole2.urdf"
        name = "cartpole"  # actor name
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = (
            False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        )
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class rewards:
        class scales:
            # reward scale for each term
            # reward_all = sum_i(terms[i] * scales[i])
            termination = -2.0
            joint_angle = -1.0
            joint_velocity = -0.005

    class normalization:
        class obs_scales:
            # 对观测值进行缩放，这里设置为1.0，即不缩放
            # 有时候需要缩放，使得神经网络的输入在一个合理的范围内，这里不需要，所以设置为1.0
            dof_pos = 1.0
            dof_vel = 1.0

        clip_observations = 5.0  # 观测值的限制范围
        clip_actions = 3000.0  # 动作的限制范围

    # viewer camera:
    class viewer:
        # 设置相机初始的位置和朝向
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]

    class sim:
        # 设置物理引擎的参数
        dt = 0.01
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            # 设置物理引擎的参数，不太需要修改，具体请参阅isaacgym的文档
            # https://blog.zzshub.cn/legged_gym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
            num_threads = 2
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class Cartpole2ConfigPPO(BaseConfig):
    seed = 1  # 设置训练随机种子
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0  # 设置初始化的噪声标准差
        actor_hidden_dims = [32, 32]  # 设置actor神经网络的隐藏层神经元数量
        critic_hidden_dims = [32, 32]  # 设置critic神经网络的隐藏层神经元数量
        # 设置激活函数，这里设置为elu
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        # 设置ppo算法的参数，具体请参阅ppo算法的论文，其实通常来说不需要修改
        # https://arxiv.org/abs/1707.06347
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        # 设置训练的迭代次数为200次
        # NOTE: 请根据实际情况修改
        max_iterations = 200  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "cartpole2"  # 实验名称，# NOTE: 用于保存模型，请注意修改！
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
