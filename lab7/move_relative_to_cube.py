#!/usr/bin/env python3

'''
This is starter code for Lab 7.

'''

import cozmo
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps
import math
import time
import sys

from odometry import cozmo_go_to_pose, my_go_to_pose3
sys.path.insert(0, '../lab6')
from pose_transform import get_relative_pose

def move_relative_to_cube(robot: cozmo.robot.Robot):

	'''Looks for a cube while sitting still, when a cube is detected it
	moves the robot to a given pose relative to the detected cube pose.'''

	# ####
	# TODO: Make the robot move to the given desired_pose_relative_to_cube.
	# Use the get_relative_pose function your implemented to determine the
	# desired robot pose relative to the robot's current pose and then use
	# one of the go_to_pose functions you implemented in Lab 6.
	# ####

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube = None
	cubeRelToRobot = None
	while cube is None:
		cube = robot.world.wait_for_observed_light_cube(timeout=30)
		if cube:
			cubeRelToRobot = get_relative_pose(cube.pose, robot.pose)
			print("Found a cube, pose in the robot coordinate frame: %s" % cubeRelToRobot)


	desired_pose_relative_to_cube = Pose(0, 0, 0, angle_z=degrees(90))#robot will go where the cube is
	pose_rel_robot = cubeRelToRobot.define_pose_relative_this(desired_pose_relative_to_cube)
	#this will always go to 2 as the cube has to be infront to be seen
	my_go_to_pose3(robot,pose_rel_robot.position.x,pose_rel_robot.position.y,pose_rel_robot.rotation.angle_z.degrees)


if __name__ == '__main__':

	cozmo.run_program(move_relative_to_cube)
