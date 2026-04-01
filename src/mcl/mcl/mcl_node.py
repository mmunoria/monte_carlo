import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from PIL import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import numpy as np
from math import sqrt, pi, exp
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def get_map(yaml_path: str):

    try:
        with open(yaml_path, "r") as f:
            info = yaml.safe_load(f)

        image_path = info["image"]
        resolution = float(info["resolution"])
        origin = info["origin"]   # [x0, y0, yaw]

        img = Image.open("/home/progress/robotics/MCL/maps/"+image_path)
        grid = np.array(img)

        return grid, resolution, origin
    
    except Exception as e:
        print(e)

def world_to_map(x, y, origin, resolution, height):

    og_x, og_y,_ = origin

    mx = int((x - og_x) / resolution)
    my = int((y - og_y) / resolution)

    #fliping image coordinates

    my_img = height - 1 - my

    return mx, my_img

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

        self.declare_parameter("target_distance", 1.0)
        self.declare_parameter("target_angle", 0.0)
        self.declare_parameter("angle_window", 0.1)

        self.target_distance = float(self.get_parameter("target_distance").value)
        self.target_angle = float(self.get_parameter("target_angle").value)
        self.angle_window = float(self.get_parameter("angle_window").value)


        self.map_loc = "/home/progress/robotics/MCL/maps/map_jackal_sim.yaml"
        self.grid, self.resolution, self.origin =  get_map(self.map_loc)
        self.r_max = 10.0
        self.step = 0.05
        self.h, self.w = self.grid.shape

        self.create_subscription(LaserScan, "/scan", self.handle_scans, sub_qos)


    def handle_scans(self, msg:LaserScan):

        index = int((self.target_angle - msg.angle_min) / msg.angle_increment)
        half = int((self.angle_window * 0.5) / msg.angle_increment)
        i0 = max(index - half, 0)
        i1 = min(index + half, len(msg.ranges) - 1)

        window_ranges = np.array(msg.ranges[i0:i1+1], dtype=float)
        valid = np.isfinite(window_ranges)
        valid &= (window_ranges >= msg.range_min) & (window_ranges <= msg.range_max)
        self.measured_values = window_ranges[valid]

        for z in self.measured_values:
            meas_error = float(z - self.target_distance)
            
            self.range_error_pub.publish(Float64(data=meas_error))

            self.n += 1
            delta = z - self.running_mean
            self.running_mean += delta / self.n
            delta2 = z - self.running_mean
            self.M2 += delta * delta2

            # update sigma_hit = sqrt(M2 / n)
            if self.n > 1:
                self.sigma_hit = sqrt(self.M2 / self.n)

            # increment outlier_count if outlier is  > 3 * sigma_hit
            if self.n > 20 and abs(z - self.running_mean) > 3 * self.sigma_hit:
                self.outlier_count += 1

            # log current estimated mean, sigma_hit, and outlier count every 25 scans
            if self.n % 25 == 0:
                self.get_logger().info(f"Scans: {self.n}  Mean: {self.running_mean:.4f} m  Sigma_hit: {self.sigma_hit:.4f} m  Outliers: {self.outlier_count}")

        # publish sigma_hit on /calibration/statistics
        self.statistics_pub.publish(Float64(data=self.sigma_hit))


        self.mean = self.running_mean
        self.error = self.mean - self.target_distance
        self.max_range = msg.range_max
       






    def cast_ray(self,px, py, theta, beam_angle):

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


    def lidar_likelihood(self, z_pred, sigma=0.2):
        err = self.z - z_pred

        return np.exp(-0.5 * np.sum((err / sigma)**2 ))




        





def main():
    rclpy.init()
    node = MCL()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
