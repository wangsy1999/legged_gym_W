from legged_gym.envs.base.base_config import BaseConfig


class Cartpole2Config(BaseConfig):
    class env:
        num_envs = 4096  # number of environments (agents) to run in parallel
        num_observations = 6  # number of observations per agent (state)
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 1  # number of actions per agent (control output)
        env_spacing = 3.0  # not used with heightfields/trimeshes
        episode_length_s = 20  # episode length in seconds

    class assets:
        fix_base_link = True  # true for fixed base link, false for free floating base link
        clip_observations = 5.0  # clip observations
        clip_actions = 1.0  # clip actions

    class rewards:
        class scales:
            termination = -2.0  # reward scale for termination
            cart_velocity = -0.01  # reward scale for cart velocity
            pole1_velocity = -0.005  # reward scale for pole1 velocity
            pole2_velocity = -0.005  # reward scale for pole2 velocity

    class viewer:
        ref_env = 0  # reference environment for the viewer
        pos = [10, 0, 6]  # [m] position of the camera
        lookat = [11.0, 5, 3.0]  # [m] point the camera is looking at

    class control:
        action_scale = 400.0  # action scale
        decimation = 1  # control decimation
        resetDist = 3.0  # reset distance

    class sim:
        dt = 0.0166
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
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
    seed = 123
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0  # initial noise std
        actor_hidden_dims = [32, 32]  # hidden dimensions of the actor network
        critic_hidden_dims = [32, 32]  # hidden dimensions of the critic network
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-4  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
