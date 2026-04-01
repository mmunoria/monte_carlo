import os
import yaml
import rclpy
import numpy as np

from PIL import Image
from math import pi, cos, sin

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_quaternion(angle: float):
    qz = np.sin(angle * 0.5)
    qw = np.cos(angle * 0.5)
    return (0.0, 0.0, qz, qw)


def get_map(yaml_path: str):
    with open(yaml_path, "r") as f:
        info = yaml.safe_load(f)

    image_path = info["image"]
    resolution = float(info["resolution"])
    origin = info["origin"]

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    abs_image_path = os.path.join(yaml_dir, image_path)

    img = Image.open(abs_image_path)
    grid = np.array(img)

    return grid, resolution, origin


def world_to_map(x, y, origin, resolution, height):
    og_x, og_y, _ = origin

    mx = int((x - og_x) / resolution)
    my = int((y - og_y) / resolution)

    # Flip image coordinates because image row 0 is top
    my_img = height - 1 - my
    return mx, my_img


def map_to_world(mx, my_img, origin, resolution, height):
    og_x, og_y, _ = origin

    my = height - 1 - my_img
    x = og_x + (mx + 0.5) * resolution
    y = og_y + (my + 0.5) * resolution
    return x, y


def is_occupied(grid, mx, my, occ_threshold=250):
    h, w = grid.shape

    if mx < 0 or mx >= w or my < 0 or my >= h:
        return True

    pixel = grid[my, mx]
    return pixel < occ_threshold


class MCL(Node):
    def __init__(self):
        super().__init__("mcl_node")

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.M = 1000
        self.is_initialized = False

        self.last_time = None

        self.u = np.zeros(2, dtype=float)              # [v_cmd, w_cmd]
        self.x_est = np.zeros(5, dtype=float)          # [px, py, theta, v, w]
        self.P = np.eye(5, dtype=float) * 1e-3

        self.particles = None                          # shape: (M, 5)
        self.weights = None                            # shape: (M,)

        self.z = None                                  # filtered lidar ranges
        self.beam_angles = None                        # filtered beam angles

        self.alpha_v = 8.0
        self.alpha_w = 10.0

        self.declare_parameter("target_angle", 0.0)
        self.declare_parameter("angle_window", 0.5)
        self.declare_parameter("sigma_lidar", 0.20)
        self.declare_parameter("r_max", 10.0)
        self.declare_parameter("ray_step", 0.05)

        self.target_angle = float(self.get_parameter("target_angle").value)
        self.angle_window = float(self.get_parameter("angle_window").value)
        self.sigma_lidar = float(self.get_parameter("sigma_lidar").value)
        self.r_max = float(self.get_parameter("r_max").value)
        self.step = float(self.get_parameter("ray_step").value)

        self.map_loc = "/home/progress/robotics/MCL/maps/map_jackal_sim.yaml"
        self.grid, self.resolution, self.origin = get_map(self.map_loc)
        self.h, self.w = self.grid.shape

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.create_subscription(LaserScan, "/scan", self.handle_scans, sub_qos)
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, "/pf/odom", 10)
        self.path_pub = self.create_publisher(Path, "/pf/path", 10)
        self.pose_pub = self.create_publisher(PoseArray, "/pf/poseArray", 10)

        self.create_timer(0.05, self.particle_filter)   # 20 Hz

    def cmd_callback(self, msg: TwistStamped):
        v_cmd = float(msg.twist.linear.x)
        w_cmd = float(msg.twist.angular.z)
        self.u = np.array([v_cmd, w_cmd], dtype=float)

    def handle_scans(self, msg: LaserScan):
        center_index = int((self.target_angle - msg.angle_min) / msg.angle_increment)
        half = int((self.angle_window * 0.5) / msg.angle_increment)

        i0 = max(center_index - half, 0)
        i1 = min(center_index + half, len(msg.ranges) - 1)

        window_ranges = np.array(msg.ranges[i0:i1 + 1], dtype=float)
        window_angles = msg.angle_min + np.arange(i0, i1 + 1) * msg.angle_increment

        valid = np.isfinite(window_ranges)
        valid &= (window_ranges >= msg.range_min) & (window_ranges <= msg.range_max)

        self.z = window_ranges[valid]
        self.beam_angles = window_angles[valid]

    def cast_ray(self, px, py, theta, beam_angle):
        angle = theta + beam_angle
        r = 0.0

        while r <= self.r_max:
            x = px + r * cos(angle)
            y = py + r * sin(angle)

            mx, my = world_to_map(x, y, self.origin, self.resolution, self.h)

            if is_occupied(self.grid, mx, my):
                return r

            r += self.step

        return self.r_max

    def h_map(self, x, beam_angles):
        px, py, theta = x[0], x[1], x[2]
        z_pred = []

        for a in beam_angles:
            z_pred.append(self.cast_ray(px, py, theta, a))

        return np.array(z_pred, dtype=float)

    def lidar_likelihood(self, z_pred, sigma=0.2):
        if self.z is None or z_pred is None:
            return 0.0
        if len(self.z) == 0 or len(z_pred) == 0:
            return 0.0
        if len(self.z) != len(z_pred):
            return 0.0

        err = self.z - z_pred
        return float(np.exp(-0.5 * np.sum((err / sigma) ** 2)))

    def f_x(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        px, py, theta, v, w = x
        v_cmd, w_cmd = u

        px_new = px + v * dt * cos(theta)
        py_new = py + v * dt * sin(theta)
        theta_new = wrap_angle(theta + w * dt)
        v_new = v + self.alpha_v * dt * (v_cmd - v)
        w_new = w + self.alpha_w * dt * (w_cmd - w)

        return np.array([px_new, py_new, theta_new, v_new, w_new], dtype=float)

    def sample_free_space_particles(self):
        free_cells = np.argwhere(self.grid >= 250)
        if len(free_cells) == 0:
            raise RuntimeError("No free cells found in map.")

        idx = np.random.choice(len(free_cells), size=self.M, replace=True)
        chosen = free_cells[idx]

        particles = np.zeros((self.M, 5), dtype=float)

        for i, (my_img, mx) in enumerate(chosen):
            px, py = map_to_world(mx, my_img, self.origin, self.resolution, self.h)
            particles[i, 0] = px
            particles[i, 1] = py
            particles[i, 2] = np.random.uniform(-pi, pi)
            particles[i, 3] = np.random.normal(self.u[0], 0.1)
            particles[i, 4] = np.random.normal(self.u[1], 0.1)

        weights = np.ones(self.M, dtype=float) / self.M
        return particles, weights

    def resample(self, particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        s = np.sum(weights)
        if s <= 0.0 or not np.isfinite(s):
            weights = np.ones(len(weights), dtype=float) / len(weights)
        else:
            weights = weights / s

        indices = np.random.choice(np.arange(len(particles)), size=len(particles), p=weights)
        return particles[indices]

    def estimate_state(self):
        self.x_est = np.average(self.particles, axis=0, weights=self.weights)

        sin_mean = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        self.x_est[2] = np.arctan2(sin_mean, cos_mean)

        diff = self.particles - self.x_est
        diff[:, 2] = np.array([wrap_angle(a) for a in diff[:, 2]])

        self.P = np.zeros((5, 5), dtype=float)
        for i in range(self.M):
            d = diff[i].reshape(5, 1)
            self.P += self.weights[i] * (d @ d.T)

    def particle_filter_step(self, dt: float):
        if self.z is None or self.beam_angles is None:
            return
        if len(self.z) == 0:
            return

        if not self.is_initialized:
            self.particles, self.weights = self.sample_free_space_particles()
            self.is_initialized = True

        new_particles = np.zeros_like(self.particles)
        new_weights = np.zeros(self.M, dtype=float)

        for i in range(self.M):
            x_pred = self.f_x(self.particles[i], self.u, dt)

            process_noise = np.array([
                np.random.normal(0.0, 0.02),
                np.random.normal(0.0, 0.02),
                np.random.normal(0.0, 0.01),
                np.random.normal(0.0, 0.05),
                np.random.normal(0.0, 0.05),
            ], dtype=float)

            x_pred = x_pred + process_noise
            x_pred[2] = wrap_angle(x_pred[2])

            z_pred = self.h_map(x_pred, self.beam_angles)
            w_map = self.lidar_likelihood(z_pred, sigma=self.sigma_lidar)

            new_particles[i] = x_pred
            new_weights[i] = w_map

        s = np.sum(new_weights)
        if s <= 0.0 or not np.isfinite(s):
            new_weights = np.ones(self.M, dtype=float) / self.M
        else:
            new_weights /= s

        self.particles = new_particles
        self.weights = new_weights

        self.estimate_state()

        self.particles = self.resample(self.particles, self.weights)
        self.weights = np.ones(self.M, dtype=float) / self.M

    def publish_particles(self, stamp):
        if self.particles is None:
            return

        msg = PoseArray()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = "odom"

        for p in self.particles:
            px, py, theta, _, _ = p
            pose = Pose()
            pose.position.x = float(px)
            pose.position.y = float(py)
            pose.position.z = 0.0

            qx, qy, qz, qw = to_quaternion(theta)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw

            msg.poses.append(pose)

        self.pose_pub.publish(msg)

    def publish_odometry(self, stamp):
        px, py, theta, v_f, w_f = self.x_est
        qx, qy, qz, qw = to_quaternion(theta)

        cov6 = np.zeros((6, 6), dtype=float)
        cov6[0, 0] = self.P[0, 0]
        cov6[0, 1] = self.P[0, 1]
        cov6[1, 0] = self.P[1, 0]
        cov6[1, 1] = self.P[1, 1]
        cov6[5, 5] = self.P[2, 2]

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(px)
        odom.pose.pose.position.y = float(py)
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance = cov6.reshape(-1).tolist()
        odom.twist.twist.linear.x = float(v_f)
        odom.twist.twist.angular.z = float(w_f)

        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose

        self.path_msg.header.stamp = odom.header.stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def particle_filter(self):
        t_now = self.get_clock().now()
        t = t_now.nanoseconds * 1e-9

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        if dt <= 0.0:
            return

        self.particle_filter_step(dt)

        if self.is_initialized:
            self.publish_odometry(t_now)
            self.publish_particles(t_now)


def main():
    rclpy.init()
    node = MCL()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()