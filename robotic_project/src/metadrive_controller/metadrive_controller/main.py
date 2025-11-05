import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from cv_bridge import CvBridge
import cv2

import numpy as np
from rclpy.wait_for_message import wait_for_message
import ros2_numpy

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC


np.random.seed(0)

class RLmetadrive(Node):

	def __init__(self):
		super().__init__('metadrive_controller')

		self.bridge = CvBridge()
		
		qos_profile_best_effort = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1
		)
		qos_profile_reliable = QoSProfile(
			reliability=QoSReliabilityPolicy.RELIABLE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1
		)
		
		self.model = PPO.load("/home/trueilorp/metadrive/bridges/ros_bridge/models/ppo_metadrive_multimodal.zip")
		# self.model = DDPG.load("/home/trueilorp/metadrive/bridges/ros_bridge/models/ddpg_metadrive_multimodal.zip")
		# self.model = SAC.load("/home/trueilorp/metadrive/bridges/ros_bridge/models/sac_metadrive_multimodal.zip")
		# print(self.model.observation_space)
		
		self.img_shape = 84
		self.object_shape = self.model.observation_space["state"].shape[0]
		
		# self.object subscription è BoundingBox3DArray perchè lo vedo su obj_bridge.py
		self.camera_sub = self.create_subscription(Image, 'metadrive/image', self.camera_callback, qos_profile_best_effort)
		self.object_sub = self.create_subscription(BoundingBox3DArray, 'metadrive/object', self.object_callback, qos_profile_best_effort) # "state" è un vettore NumPy lungo 19 o 80 elementi. 
		# l'ordine tipico di 'state': x, y, z, size_x, size_y, size_z + eventuali padding.
		# self.lidar_sub = self.create_subscription(PointCloud2, 'metadrive/lidar', self.lidar_callback, qos_profile_best_effort)

		self.camera_msg = None
		self.object_msg = None
		# self.lidar_msg = None
		
		self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', qos_profile_reliable)

		wait_for_message(Image, self, 'metadrive/image')
		wait_for_message(BoundingBox3DArray, self, 'metadrive/object')
		# wait_for_message(PointCloud2, self, 'metadrive/lidar')

		self.dt = 0.1
		self.timer = self.create_timer(self.dt, self.control_loop)

	def camera_callback(self, msg):
		"""Callback per convertire il messaggio ROS Image in un frame numpy compatibile col modello."""
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') # converto il messaggio di ROS2 in img opencv
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # converto in RGB
			if cv_image.shape[:2] != (84, 84): # ridimensiono l'immagine a (84,84) se non lo è già
				cv_image = cv2.resize(cv_image, (84, 84), interpolation=cv2.INTER_AREA)
			cv_image = cv_image.astype(np.float32) / 255.0 # normalizzo tra 0 e 1
			cv_image = np.expand_dims(cv_image, axis=-1) # expando perchè nel training trainavo img con shape (84,84,3,1)
			# cv_image = np.transpose(cv_image, (3, 2, 0, 1)) # riordino in (84,84,3,1)
			self.camera_msg = cv_image
		except Exception as e:
			self.get_logger().error(f"Errore nella conversione dell'immagine: {e}", exc_info=True)

				
	def object_callback(self, msg):
		object_list = msg.boxes
		all_detected_objects = [] # lista vuota per contenere i dati di tutti gli oggetti
		for box in object_list: # per ogni oggetto nella lista (pedoni, ...)
			center_pos = box.center.position # estrapolo la pos x,y,z (con z=0.0)
			center_orient = box.center.orientation # estrapolo la rot (quaternion4D) con x=0.0 e y=0.0
			box_size = box.size # estrapolo la size (lunghezza, larghezza, altezza) dell'oggetto
			all_detected_objects.append({
				"position": center_pos,
				"orientation": center_orient,
				"size": box_size
			}) # aggiungo il dizionario alla lista
		self.object_msg = all_detected_objects
		#self.get_logger().info(f'OBJECT --> Ricevuti e processati {len(self.object_msg)} oggetti.')
	

	# def lidar_callback(self, msg: PointCloud2):
	# 	"""
	# 	Callback finale e funzionante per il LiDAR di MetaDrive.
	# 	Estrae i dati dalla chiave 'xyz' del dizionario restituito.
	# 	"""
	# 	try:
	# 		# La conversione di ros2_numpy restituisce un dizionario
	# 		data_dict = ros2_numpy.numpify(msg)
			
	# 		# LA SOLUZIONE: Usiamo la chiave corretta 'xyz' che abbiamo scoperto!
	# 		# Il valore associato è già un array NumPy (N, 3) con le coordinate.
	# 		self.lidar_msg = data_dict['xyz']
			
	# 		# Log di successo (opzionale, puoi rimuoverlo una volta che vedi che funziona)
	# 		self.get_logger().info(f"LIDAR --> estratti con successo dalla chiave 'xyz'. Shape: {self.lidar_msg.shape}")

	# 	except Exception as e:
	# 		self.get_logger().error(f"Errore imprevisto nella lidar_callback finale: {e}", exc_info=True)

	def control_loop(self):
		
		print("\n ------------------- Control Loop Started ------------------")
		
		# Get messages from metadrive
		# print("Camera MSG", self.camera_msg)
		# print("Object MSG", self.object_msg)
		# print("Lidar MSG", self.lidar_msg)
		
		# preprocess object
		if len(self.object_msg) == 0:
				object_msg = np.zeros((self.object_shape,), dtype=np.float32)
		else:
			object_msg = self.map_objects_to_state(self.object_msg)
		
		# # preprocess image, resize to the training shape (84,84)
		# if len(self.camera_msg) != 0:
		# 	img = self.camera_msg  # assuming it's a numpy array already (900x1200x3)
		# 	# img_resized = cv2.resize(img, (self.img_shape, self.img_shape), interpolation=cv2.INTER_AREA) # resize to match training dim
			
		# 	# print("Length Object MSG:", len(self.object_msg))
		# 	# Detected objects: [{'position': geometry_msgs.msg.Point(x=52.41254425048828, y=-4.6554059982299805, z=0.0), 'orientation': geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.021263538997177787, w=0.9997739053952726), 'size': geometry_msgs.msg.Vector3(x=4.869999885559082, y=2.0460000038146973, z=1.850000023841858)}]
		img = self.camera_msg
		
		# ------- BUILD OBS -------
		obs = {
			"image": img,
			"state": np.array(object_msg, dtype=np.float32)
		}
		print("Observation:", obs)
		
		# ------- PREDICT ACTION -------
		action, _ = self.model.predict(obs, deterministic=True)
		# action = action[0] # sto passando un batch di osservazioni al modello
		print("Predicted Action:", action) # Predicted Action: [-0.05352883  0.16370402]
		steering = action[0]
		throttle = action[1]
		
		print("Steering:", steering, "Throttle:", throttle)
		
		# ------- PUBLISH CMD_VEL -------
		twist = Twist()
		twist.linear.x = float(throttle)
		twist.angular.z = float(steering)
		self.publisher_vel.publish(twist)
		# print("Published Twist:", twist)
		
		print("------------------- Control Loop Executed ------------------ \n")

	def map_objects_to_state(self, object_list):
		max_objects = 3
		state = []
		
		# print("Object_list:", object_list)
		for i in range(max_objects):
			if i < len(object_list):
				obj = object_list[i]
				pos = obj['position']
				size = obj['size']

				# Aggiungi x,y,z della posizione e x,y,z delle dimensioni
				state.extend([pos.x, pos.y, pos.z, size.x, size.y, size.z])
			else:
				# Se ci sono meno oggetti, riempi con zeri
				state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

		# PPO era stato allenato con 19 elementi → aggiungi un valore finale (ad esempio 0)
		if len(state) < self.object_shape:
			state.append(0.0)

		# Converti in numpy array float32
		return np.array(state[:self.object_shape], dtype=np.float32)
	
	def cleanup(self):
		# Stop signal
		twist = Twist()
		twist.linear.x = 0.0
		twist.angular.z = 0.0
		self.publisher_vel.publish(twist)
		self.get_logger().info("Sent stop Twist before shutdown.")

		# Force ROS executor to send out messages
		rclpy.spin_once(self, timeout_sec=0.2)

		# Destroy publisher completely
		self.publisher_vel.destroy()
		self.get_logger().info("Destroyed publisher to flush QoS queue.")

		# Kill timers, cleanup
		self.destroy_timer(self.timer)
	
def main(args=None):
	rclpy.init(args=args)
	rl_meta = RLmetadrive()
	try:
		rclpy.spin(rl_meta)
	except KeyboardInterrupt:
		rl_meta.get_logger().info("Keyboard interrupt — shutting down controller...")
	finally:
		rl_meta.cleanup()
		rl_meta.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()