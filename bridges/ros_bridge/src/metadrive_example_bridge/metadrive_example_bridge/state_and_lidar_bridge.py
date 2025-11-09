import struct
import numpy as np
import rclpy
import zmq
from builtin_interfaces.msg import Time
from rclpy.node import Node
from std_msgs.msg import Header, Float32MultiArray


class StateAndLidarPublisher(Node):
	def __init__(self):
		super().__init__('state_and_lidar_publisher')

		# Publisher per l'array di float (stato + lidar)
		self.publisher_ = self.create_publisher(
			Float32MultiArray,
			'metadrive/state_and_lidar',
			qos_profile=10
		)

		# ZMQ: riceve i dati inviati dal simulatore
		context = zmq.Context().instance()
		self.socket = context.socket(zmq.PULL)
		self.socket.setsockopt(zmq.CONFLATE, 1)
		self.socket.set_hwm(5)
		self.socket.connect("ipc:///tmp/state_and_lidar")  # stesso nome usato nel simulatore

		# Timer per leggere periodicamente la socket
		self.timer_period = 0.05  # 20 Hz
		self.timer = self.create_timer(self.timer_period, self.timer_callback)
		self.i = 0

	def timer_callback(self):
		try:
			# Leggi il messaggio dalla socket
			msg = self.socket.recv(flags=zmq.NOBLOCK)

			# Primo int32 = lunghezza
			length = struct.unpack('i', msg[:4])[0]

			# Il resto è l'array di float32
			data = np.frombuffer(msg[4:], dtype=np.float32)
			if len(data) != length:
				self.get_logger().warn(f"Lunghezza mismatch: header={length}, data={len(data)}")

			# Crea messaggio ROS
			msg_ros = Float32MultiArray()
			msg_ros.data = data.tolist()

			# Pubblica
			self.publisher_.publish(msg_ros)

			self.i += 1

		except zmq.Again:
			# Nessun messaggio disponibile (socket vuota)
			pass
		except Exception as e:
			self.get_logger().error(f"Error nel ricevere state_and_lidar: {e}", exc_info=True)

	def get_msg_header(self):
		"""
		(Non strettamente necessario qui, ma lo manteniamo per consistenza)
		"""
		header = Header()
		header.frame_id = "map"
		t = self.get_clock().now()
		sec, nsec = t.seconds_nanoseconds()
		from builtin_interfaces.msg import Time
		time = Time(sec=sec, nanosec=nsec)
		header.stamp = time
		return header


def main(args=None):
	rclpy.init(args=args)
	node = StateAndLidarPublisher()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
