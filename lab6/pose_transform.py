#!/usr/bin/env python3

'''
This is starter code for Lab 6 on Coordinate Frame transforms.

'''

import asyncio
import cozmo
import numpy
from cozmo.util import degrees

def get_relative_pose(object_pose, refrence_frame_pose):
	# ####
	# TODO: Implement computation of the relative frame using numpy.
	# Try to derive the equations yourself and verify by looking at
	# the books or slides before implementing.
	# ####

	objectx,objecty,objectz = object_pose.position.x_y_z
	objectAngle = object_pose.rotation.angle_z

	refrence_framex, refrence_framey, refrence_framez = refrence_frame_pose.position.x_y_z
	refrence_frameAngle = refrence_frame_pose.rotation.angle_z

	sinObjAngle = numpy.sin(objectAngle.radians)
	cosObjAngle = numpy.cos(objectAngle.radians)

	newx = objectx + (cosObjAngle * refrence_framex) - (sinObjAngle * refrence_framey)
	newy = objecty + (sinObjAngle * refrence_framex) + (cosObjAngle * refrence_framey)
	newz = objectz + refrence_framez
	newAngle = objectAngle + refrence_frameAngle

	return cozmo.util.Pose(newx,newy,newz,angle_z=newAngle,origin_id = object_pose._origin_id)

def find_relative_cube_pose(robot: cozmo.robot.Robot):
	'''Looks for a cube while sitting still, prints the pose of the detected cube
	in world coordinate frame and relative to the robot coordinate frame.'''

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube = None

	while True:
		try:
			cube = robot.world.wait_for_observed_light_cube(timeout=30)
			if cube:
				print("Robot pose: %s" % robot.pose)
				print("Cube pose: %s" % cube.pose)
				print("Cube pose in the robot coordinate frame: %s" % get_relative_pose(cube.pose, robot.pose))
		except asyncio.TimeoutError:
			print("Didn't find a cube")


if __name__ == '__main__':

	cozmo.run_program(find_relative_cube_pose)
