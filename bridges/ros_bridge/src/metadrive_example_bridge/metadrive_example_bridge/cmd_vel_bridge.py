import struct
import zmq
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class CmdVelSubscriber(Node):

	def __init__(self):
		super().__init__('cmdvel_subscriber')
		
		qos_profile_best_effort = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1
		)
		
		# ---- ROS ----
		self.subscription = self.create_subscription(
			Twist,
			'/cmd_vel',
			self.cmd_vel_callback,
			qos_profile_best_effort)
		self.subscription  # prevent unused variable warning

		# ---- ZeroMQ ----
		context = zmq.Context().instance()
		context.setsockopt(zmq.IO_THREADS, 2)
		self.socket = context.socket(zmq.PUSH)
		self.socket.setsockopt(zmq.SNDBUF, 4194304)
		self.socket.connect("ipc:///tmp/cmd_vel")  # deve corrispondere al socket aperto in MetaDrive

	def cmd_vel_callback(self, msg: Twist):
		"""
		Callback ROS che invia linear.x e angular.z via ZMQ a MetaDrive
		"""
		# impacchetta i due float (steering, throttle)
		data = struct.pack('ff', msg.angular.z, msg.linear.x)
		try:
			self.socket.send(data, zmq.NOBLOCK)
		except zmq.error.Again:
			self.get_logger().warn("Non riesco a mandare cmd_vel, buffer pieno!")


def main(args=None):
	rclpy.init(args=args)
	node = CmdVelSubscriber()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
