#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

path = Path()
odom = Odometry()
def path_callback(msg):
	path = msg
	for pose in path.poses:
		print(pose.pose.position.x, pose.pose.position.y)
	print("=============================================================")
def odom_callback(msg):
	odom = msg


if __name__ == '__main__':
	rospy.init_node('planner_node')
	rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, path_callback)
	rospy.Subscriber('/odom', Odometry, odom_callback)
	
	rospy.spin()