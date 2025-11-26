import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    """Generate random float tensor in range [lower, upper]"""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class MobileRobotEnv:
    """
    Mobile robot navigation environment for Genesis RL training.

    Simplified from go2_env.py for 2D navigation tasks.
    - 3 DOF actions: [vx, vy, omega] (linear x, linear y, angular z)
    - Simple observation space: base velocity + goal position
    - Navigation rewards: goal tracking, smooth motion, collision avoidance
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # 3: vx, vy, omega
        self.num_commands = command_cfg["num_commands"]  # 2: goal_x, goal_y
        self.device = gs.device

        self.dt = 0.02  # 50Hz control frequency
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(5.0, 0.0, 8.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=False,  # No joints for mobile base
                max_collision_pairs=10,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add mobile robot base
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # For now, use a simple box as mobile robot (can be replaced with URDF later)
        # TODO: Replace with actual mobile robot URDF when available
        self.robot = self.scene.add_entity(
            gs.morphs.Box(
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                size=(0.5, 0.4, 0.2),  # 50cm x 40cm x 20cm mobile base
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        # Goal positions (commands)
        self.goals = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_goals(self, envs_idx):
        """Randomly sample new goal positions for given environments"""
        self.goals[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["goal_x_range"], (len(envs_idx),), gs.device
        )
        self.goals[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["goal_y_range"], (len(envs_idx),), gs.device
        )

    def step(self, actions):
        """
        Execute one step of the environment.

        Args:
            actions: (num_envs, 3) tensor of [vx, vy, omega] commands

        Returns:
            observations, rewards, dones, extras
        """
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # Apply velocity commands directly to the robot
        # Scale actions to real velocity ranges
        target_lin_vel = self.actions[:, :2] * self.env_cfg["action_scale_linear"]  # vx, vy
        target_ang_vel = self.actions[:, 2:3] * self.env_cfg["action_scale_angular"]  # omega

        # Set velocities directly (Genesis will handle physics)
        # Note: For more realistic control, we'd use force/torque control
        self.robot.set_vel(
            torch.cat([target_lin_vel, torch.zeros((self.num_envs, 1), device=gs.device)], dim=1)
        )
        self.robot.set_ang(
            torch.cat([torch.zeros((self.num_envs, 2), device=gs.device), target_ang_vel], dim=1)
        )

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # resample goals periodically
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_goals(envs_idx)

        # check termination and reset
        # Goal reached: within threshold distance
        goal_distance = torch.sqrt(
            (self.base_pos[:, 0] - self.goals[:, 0]) ** 2 +
            (self.base_pos[:, 1] - self.goals[:, 1]) ** 2
        )
        goal_reached = goal_distance < self.env_cfg["goal_threshold"]

        # Timeout
        timeout = self.episode_length_buf > self.max_episode_length

        # Collision/fall detection (z position too low)
        collision = self.base_pos[:, 2] < self.env_cfg["min_height"]

        self.reset_buf = goal_reached | timeout | collision

        time_out_idx = timeout.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # Track goal success
        goal_reached_idx = goal_reached.nonzero(as_tuple=False).reshape((-1,))
        self.extras["goal_reached"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["goal_reached"][goal_reached_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        # Observation space: base velocity (3) + goal position relative (2) + distance to goal (1) + actions (3) = 9
        goal_relative = self.goals - self.base_pos[:, :2]
        goal_distance_scaled = goal_distance.unsqueeze(1) * self.obs_scales["goal_dist"]

        self.obs_buf = torch.cat(
            [
                self.base_lin_vel[:, :2] * self.obs_scales["lin_vel"],  # 2: vx, vy
                self.base_ang_vel[:, 2:3] * self.obs_scales["ang_vel"],  # 1: omega
                goal_relative * self.obs_scales["goal_pos"],  # 2: relative goal position
                goal_distance_scaled,  # 1: distance to goal
                self.actions,  # 3: last actions
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        """Reset specific environments"""
        if len(envs_idx) == 0:
            return

        # reset base position with some randomization
        self.base_pos[envs_idx, 0] = self.base_init_pos[0] + gs_rand_float(
            -self.env_cfg["init_pos_noise"], self.env_cfg["init_pos_noise"], (len(envs_idx),), gs.device
        )
        self.base_pos[envs_idx, 1] = self.base_init_pos[1] + gs_rand_float(
            -self.env_cfg["init_pos_noise"], self.env_cfg["init_pos_noise"], (len(envs_idx),), gs.device
        )
        self.base_pos[envs_idx, 2] = self.base_init_pos[2]

        # random initial orientation
        yaw = gs_rand_float(-math.pi, math.pi, (len(envs_idx),), gs.device)
        self.base_quat[envs_idx, 0] = torch.cos(yaw / 2)  # w
        self.base_quat[envs_idx, 1] = 0.0  # x
        self.base_quat[envs_idx, 2] = 0.0  # y
        self.base_quat[envs_idx, 3] = torch.sin(yaw / 2)  # z

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_goals(envs_idx)

    def reset(self):
        """Reset all environments"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions ----------------
    def _reward_goal_tracking(self):
        """Reward for moving towards goal"""
        goal_distance = torch.sqrt(
            (self.base_pos[:, 0] - self.goals[:, 0]) ** 2 +
            (self.base_pos[:, 1] - self.goals[:, 1]) ** 2
        )
        # Exponential reward - higher when closer to goal
        return torch.exp(-goal_distance / self.reward_cfg["goal_tracking_sigma"])

    def _reward_goal_reached(self):
        """Bonus reward for reaching goal"""
        goal_distance = torch.sqrt(
            (self.base_pos[:, 0] - self.goals[:, 0]) ** 2 +
            (self.base_pos[:, 1] - self.goals[:, 1]) ** 2
        )
        return (goal_distance < self.env_cfg["goal_threshold"]).float()

    def _reward_action_smoothness(self):
        """Penalize rapid changes in actions"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_linear_velocity(self):
        """Penalize excessive linear velocity (encourage smooth motion)"""
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

    def _reward_angular_velocity(self):
        """Penalize excessive angular velocity (encourage smooth turning)"""
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_action_magnitude(self):
        """Penalize large actions (energy efficiency)"""
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_orientation(self):
        """Reward for facing the goal"""
        # Vector from robot to goal
        to_goal = self.goals - self.base_pos[:, :2]
        to_goal_norm = to_goal / (torch.norm(to_goal, dim=1, keepdim=True) + 1e-8)

        # Robot's forward direction (transformed by quaternion)
        forward = torch.tensor([1.0, 0.0, 0.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        robot_forward = transform_by_quat(forward, self.base_quat)[:, :2]
        robot_forward_norm = robot_forward / (torch.norm(robot_forward, dim=1, keepdim=True) + 1e-8)

        # Dot product - higher when facing goal
        alignment = torch.sum(to_goal_norm * robot_forward_norm, dim=1)
        return alignment
