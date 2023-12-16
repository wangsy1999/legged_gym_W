from .base_task import BaseTask
from isaacgym import gymapi
from abc import abstractmethod
import torch
import legged_gym.utils.helpers as helpers
import numpy as np
from typing import Dict
from typing import Any
import operator
import random
from copy import deepcopy
from legged_gym.utils.dr_utils import (
    get_property_setter_map,
    get_property_getter_map,
    get_default_setter_args,
    apply_random_samples,
    check_buckets,
    generate_random_samples,
)


EXISTING_SIM = None


# TODO: add abstract attribution decorator
class IsaacGymEnvsAdaptionLayer(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg_dict = helpers.class_to_dict(
            cfg
        )  # change cfg from python class to dict for IsaacGemEnvs adaption
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.sim_initialized = False
        self.rl_device = sim_device
        

        # set environment parameters and basic trainning parameters for IsaacGymEnvs adaption
        self.num_agents = self.cfg_dict["env"].get("numAgents", 1)
        self.num_observations = self.cfg_dict["env"]["num_observations"]
        self.num_states = self.cfg_dict["env"].get("numStates", 0)
        # TODO: add obs_space, state_space, act_space
        self.control_freq_inv = self.cfg_dict["env"].get("controlFrequencyInv", 1)
        self.clip_obs = self.cfg_dict["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.cfg_dict["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0
        # number of control steps
        self.control_steps: int = 0
        # TODO: add record frames

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.allocate_extra_buffers()
        self.dt = self.sim_params.dt

        #TODO: set camera look at
        #if not self.headless:
        #    self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # set params for randomization
        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.sim_initialized = True
        self.obs_dict = {}

    def create_sim(
        self,
        compute_device: int,
        graphics_device: int,
        physics_engine,
        sim_params: gymapi.SimParams,
    ):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        global EXISTING_SIM
        if EXISTING_SIM is not None:
            print("[warning]: gym already exist, return exist one!")
            return EXISTING_SIM
        else:
            EXISTING_SIM = self.gym.create_sim(
                compute_device=compute_device,
                graphics_device=graphics_device,
                type=physics_engine,
                params=sim_params,
            )
        if EXISTING_SIM is None:
            print("*** Failed to create sim")
            quit()

        return EXISTING_SIM

    def allocate_extra_buffers(self):
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float
        )
        self.timeout_buf = self.time_out_buf  # rename for compatibility reason
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    def get_state(self):
        state = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(
            self.rl_device
        )
        self.device
        return state

    @abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    @abstractmethod
    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx.
        Should be implemented in an environment class inherited from VecTask.
        """

    def step(self, actions: torch.Tensor):
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, privileged observations, rewards, resets, info
        """
        # return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # apply actions
        self.pre_physics_step(actions=action_tensor)

        self.render()  # FIXME: check render issue
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)  # created by create sim
            # TODO: rewrite pd controller and investigate control_freq_inv

        # to fix! [comments from isaacgymenv]
        # if self.device == "cpu":
        #    self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )
        # TODO: add support for privileged observation randomization
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)
        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        # return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        return self.obs_dict["obs"], None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_dict["obs"]

    def get_privileged_observations(self):
        return None  # TODO: add support for privileged observation

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros(
            [self.num_envs, self.num_actions],
            dtype=torch.float32,
            device=self.rl_device,
        )

        return actions

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == "color":
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + "_" + str(prop_idx) + "_" + attr
                        lo_hi = attr_randomization_params["range"]
                        distr = attr_randomization_params["distribution"]
                        if "uniform" not in distr:
                            lo_hi = (-1.0 * float("Inf"), float("Inf"))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + "_" + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(
                self.randomize_buf >= rand_freq,
                torch.ones_like(self.randomize_buf),
                torch.zeros_like(self.randomize_buf),
            )
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = (
                    dr_params[nonphysical_param]["schedule"]
                    if "schedule" in dr_params[nonphysical_param]
                    else None
                )
                sched_step = (
                    dr_params[nonphysical_param]["schedule_steps"]
                    if "schedule" in dr_params[nonphysical_param]
                    else None
                )
                op = operator.add if op_type == "additive" else operator.mul

                if sched_type == "linear":
                    sched_scaling = 1.0 / sched_step * min(self.last_step, sched_step)
                elif sched_type == "constant":
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == "gaussian":
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get(
                        "range_correlated", [0.0, 0.0]
                    )

                    if op_type == "additive":
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == "scaling":
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * (
                            1.0 - sched_scaling
                        )  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * (
                            1.0 - sched_scaling
                        )  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get("corr", None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params["corr"] = corr
                        corr = corr * params["var_corr"] + params["mu_corr"]
                        return op(
                            tensor,
                            corr
                            + torch.randn_like(tensor) * params["var"]
                            + params["mu"],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        "mu": mu,
                        "var": var,
                        "mu_corr": mu_corr,
                        "var_corr": var_corr,
                        "noise_lambda": noise_lambda,
                    }

                elif dist == "uniform":
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get(
                        "range_correlated", [0.0, 0.0]
                    )

                    if op_type == "additive":
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == "scaling":
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get("corr", None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params["corr"] = corr
                        corr = (
                            corr * (params["hi_corr"] - params["lo_corr"])
                            + params["lo_corr"]
                        )
                        return op(
                            tensor,
                            corr
                            + torch.rand_like(tensor) * (params["hi"] - params["lo"])
                            + params["lo"],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        "lo": lo,
                        "hi": hi,
                        "lo_corr": lo_corr,
                        "hi_corr": hi_corr,
                        "noise_lambda": noise_lambda,
                    }

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)
                }

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop,
                    self.original_props["sim_params"],
                    attr,
                    attr_randomization_params,
                    self.last_step,
                )

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props
        # and lastly loop over the ranges of the params

        for actor, actor_properties in dr_params["actor_params"].items():
            # Loop over all envs as this part is not tensorised yet
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties
                #          prop_attrs:
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == "color":
                        num_bodies = self.gym.get_actor_rigid_body_count(env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(
                                env,
                                handle,
                                n,
                                gymapi.MESH_VISUAL,
                                gymapi.Vec3(
                                    random.uniform(0, 1),
                                    random.uniform(0, 1),
                                    random.uniform(0, 1),
                                ),
                            )
                        continue

                    if prop_name == "scale":
                        setup_only = prop_attrs.get("setup_only", False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(
                                attr_randomization_params, 1, self.last_step, None
                            )
                            og_scale = 1
                            if attr_randomization_params["operation"] == "scaling":
                                new_scale = og_scale * sample
                            elif attr_randomization_params["operation"] == "additive":
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)}
                                for p in prop
                            ]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get(
                                    "setup_only", False
                                )
                                if (
                                    setup_only and not self.sim_initialized
                                ) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        (
                                            smpl,
                                            extern_offsets[env_id],
                                        ) = get_attr_val_from_sample(
                                            extern_sample,
                                            extern_offsets[env_id],
                                            p,
                                            attr,
                                        )
                                    apply_random_samples(
                                        p,
                                        og_p,
                                        attr,
                                        attr_randomization_params,
                                        self.last_step,
                                        smpl,
                                    )
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get(
                                "setup_only", False
                            )
                            if (
                                setup_only and not self.sim_initialized
                            ) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    (
                                        smpl,
                                        extern_offsets[env_id],
                                    ) = get_attr_val_from_sample(
                                        extern_sample,
                                        extern_offsets[env_id],
                                        prop,
                                        attr,
                                    )
                                apply_random_samples(
                                    prop,
                                    self.original_props[prop_name],
                                    attr,
                                    attr_randomization_params,
                                    self.last_step,
                                    smpl,
                                )
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print(
                            "env_id",
                            env_id,
                            "extern_offset",
                            extern_offsets[env_id],
                            "vs extern_sample.shape",
                            extern_sample.shape,
                        )
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False


# rename class for compatibility
class VecTask(IsaacGymEnvsAdaptionLayer):
    pass
