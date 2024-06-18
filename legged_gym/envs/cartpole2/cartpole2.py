import sys
from isaacgym import gymapi
from isaacgym import gymutil
import torch
from legged_gym.envs.base.base_task import BaseTask
from cartpole2_config import Cartpole2Config


class Cartpole2Task(BaseTask):
    def __init__(self, cfg: Cartpole2Config, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
