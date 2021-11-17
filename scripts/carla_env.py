import rospy
import carla
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from transforms3d.euler import euler2quat
import math
from sensor_msgs.msg import Imu
from sensor_msgs.point_cloud2 import create_cloud
from threading import Thread

import ros_compatibility as roscomp
from carla_setting import *

lidar_publisher = None

class CarlaEnv:
    def __init__(self):
        # client
        self.client = carla.Client(CARLA_HOSTS[0], CARLA_HOSTS[1])
        self.client.set_timeout(CARLA_TIMEOUT)
        self.client.load_world(CARLA_MAP)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        print(settings)
        settings.fixed_delta_seconds = 0.05
        print(settings)
        self.world.apply_settings(settings)

        self.actor_list = []
        self.lidar_data = None

        # ros
        self.image_pub = rospy.Publisher(IMAGE_TOPIC, Image, queue_size=1)
        self.lidar_publisher = rospy.Publisher(LIDAR_TOPIC, PointCloud2, queue_size=1)
        self.imu_publisher = rospy.Publisher(IMU_TOPIC, Imu, queue_size=1)

        # bp set
        self.blueprint_library = self.world.get_blueprint_library()
        # vehicle
        bp = self.blueprint_library.filter(VEHICLE)[0]
        spawn_point = self.world.get_map().get_waypoint(self.world.get_map().get_spawn_points()[1].location).transform
        spawn_point.location += carla.Location(z=1.0)
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(True)

        # rgb
        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        rgb_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        rgb_bp.set_attribute('fov', f"{IMAGE_FOV}")
        spawn_point = carla.Transform(carla.Location(x=1.2, z=2.0))
        rgb_sensor = self.world.spawn_actor(rgb_bp, spawn_point, attach_to=self.vehicle)
        rgb_sensor.listen(lambda data: self.pub_img(data))
        self.actor_list.append(rgb_sensor)
        
        # lidar
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNEL))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_LOWER_FOV))
        lidar_bp.set_attribute('points_per_second',str(LIDAR_POINT_PER_SENCOD))
        lidar_bp.set_attribute('rotation_frequency',str(LIDAR_FREQUENCY))
        lidar_bp.set_attribute('dropoff_general_rate',str(0.0))
        lidar_bp.set_attribute('dropoff_intensity_limit',str(0.0))
        lidar_bp.set_attribute('dropoff_zero_intensity',str(0.0))
        lidar_bp.set_attribute('noise_stddev',str(LIDAR_NOISE_STDDEV))
        lidar_bp.set_attribute('sensor_tick',str(0.05))
        lidar_transform = carla.Transform(carla.Location(0,0,2))
        lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to = self.vehicle)
        lidar_sensor.listen(lambda data: self.lidar_handler(data))
        self.actor_list.append(lidar_sensor)
        self.thread = Thread(target=self._update_thread)
        self.thread.start()

        # # imu
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        imu_bp.set_attribute('noise_accel_stddev_x',str(0.0))
        imu_bp.set_attribute('noise_accel_stddev_y',str(0.0))
        imu_bp.set_attribute('noise_accel_stddev_z',str(0.0))
        imu_bp.set_attribute('noise_gyro_stddev_x',str(0.0))
        imu_bp.set_attribute('noise_gyro_stddev_y',str(0.0))
        imu_bp.set_attribute('noise_gyro_stddev_z',str(0.0))
        imu_bp.set_attribute('noise_gyro_bias_x',str(0.0))
        imu_bp.set_attribute('noise_gyro_bias_y',str(0.0))
        imu_bp.set_attribute('noise_gyro_bias_z',str(0.0))
        imu_transform = carla.Transform(carla.Location(2,0,2))
        imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to = self.vehicle)
        imu_sensor.listen(lambda data: self.imu_sensor_data_updated(data))
        self.actor_list.append(imu_sensor)

    def _update_thread(self):
        self.world.wait_for_tick()
        self.sensor_data_updated()

    def lidar_handler(self, data):
        self.lidar_data = data

    def sensor_data_updated(self):
        while not rospy.is_shutdown():
            if self.lidar_data != None:
                header = Header(frame_id=LIDAR_FRAME, stamp = roscomp.ros_timestamp(sec=self.lidar_data.timestamp, from_sec=True))
                lidar_data = np.fromstring(bytes(self.lidar_data.raw_data), dtype=np.float32)
                lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
                lidar_data[:, 1] *= -1
                fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1)]
                point_cloud_msg = create_cloud(header, fields, lidar_data)
                self.lidar_publisher.publish(point_cloud_msg)

    def imu_sensor_data_updated(self, carla_imu_measurement):
        imu_msg = Imu()
        # print(carla_imu_measurement)
        imu_msg.header = Header(frame_id=IMU_FRAME, stamp = roscomp.ros_timestamp(sec=carla_imu_measurement.timestamp, from_sec=True))

        imu_msg.angular_velocity.x = -carla_imu_measurement.gyroscope.x
        imu_msg.angular_velocity.y = carla_imu_measurement.gyroscope.y
        imu_msg.angular_velocity.z = -carla_imu_measurement.gyroscope.z

        imu_msg.linear_acceleration.x = carla_imu_measurement.accelerometer.x
        imu_msg.linear_acceleration.y = -carla_imu_measurement.accelerometer.y
        imu_msg.linear_acceleration.z = carla_imu_measurement.accelerometer.z

        roll = math.radians(carla_imu_measurement.transform.rotation.roll)
        pitch = -math.radians(carla_imu_measurement.transform.rotation.pitch)
        yaw = -math.radians(carla_imu_measurement.transform.rotation.yaw)

        quat = euler2quat(roll, pitch, yaw)

        imu_msg.orientation.w = quat[0]
        imu_msg.orientation.x = quat[1]
        imu_msg.orientation.y = quat[2]
        imu_msg.orientation.z = quat[3]
        self.imu_publisher.publish(imu_msg)

    def pub_img(self, image):
        bridge = CvBridge()
        img = np.array(image.raw_data).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]
        self.image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
    
    def destroy_agent(self):
        for actor in self.actor_list:
            actor.destroy()

def main():
    global img, image_pub, lidar_publisher

    rospy.init_node("carla_env")
    carla_env = CarlaEnv()
    
    # rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        rospy.spin()
        # rospy.sleep(rate)

    carla_env.destroy_agent()

if __name__ == "__main__":
    main()