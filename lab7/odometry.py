#!/usr/bin/env python3

'''
Stater code for Lab 7.

'''

import cozmo
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps
import math
import time

# Wrappers for existing Cozmo navigation functions

def cozmo_drive_straight(robot, dist, speed):
	"""Drives the robot straight.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		dist -- Desired distance of the movement in millimeters
		speed -- Desired speed of the movement in millimeters per second
	"""
	robot.drive_straight(distance_mm(dist), speed_mmps(speed)).wait_for_completed()

def cozmo_turn_in_place(robot, angle, speed):
	"""Rotates the robot in place.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle -- Desired distance of the movement in degrees
		speed -- Desired speed of the movement in degrees per second
	"""
	robot.turn_in_place(degrees(angle), speed=degrees(speed)).wait_for_completed()

def cozmo_go_to_pose(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	robot.go_to_pose(Pose(x, y, 0, angle_z=degrees(angle_z)), relative_to_robot=True).wait_for_completed()

# Functions to be defined as part of the labs

def get_front_wheel_radius():
	"""Returns the radius of the Cozmo robot's front wheel in millimeters."""
	# ####
	# TODO: Empirically determine the radius of the robot's front wheel using the
	# cozmo_drive_straight() function. You can write a separate script for doing 
	# experiments to determine the radius. This function should return the radius
	# in millimeters. Write a comment that explains how you determined it and any
	# computation you do as part of this function.
	# ####
	#c=2(pi)r, r = c/2(pi), count number of revolutions for distance x. distance/rev = c
	#he rotated 1 rotation in about 87 mm
	return 87/(2*math.pi)

def get_distance_between_wheels():
	"""Returns the distance between the wheels of the Cozmo robot in millimeters."""
	# ####
	# TODO: Empirically determine the distance between the wheels of the robot using
	# robot.drive_wheels() function. Write a comment that explains how you determined
	# it and any computation you do as part of this function.
	# ####
	#drive both wheels at different speeds until original point is reached. calculate distance traveled by each wheel from time and speed
	#distance travelled = c. c=pi*2*r, difference between 2 rs is the distance between wheels
	#100mm/s * 14.2s = 1420mm -> r=1420/2pi
	#60mm/s * 14.2 = 852 mm -> r= 852/2pi
	#b= 1420-852/2pi
	return 568/(2*math.pi)

def rotate_front_wheel(robot, angle_deg):
	"""Rotates the front wheel of the robot by a desired angle.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle_deg -- Desired rotation of the wheel in degrees
	"""
	# ####
	# TODO: Implement this function.
	# ####
	# angle/2*pi = dist/2*pi*r -> dist = angle*r
	cozmo_drive_straight(robot,get_front_wheel_radius()*math.radians(angle_deg),25)

def my_drive_straight(robot, dist, speed):
	"""Drives the robot straight.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		dist -- Desired distance of the movement in millimeters
		speed -- Desired speed of the movement in millimeters per second
	"""
	# ####
	# TODO: Implement your version of a driving straight function using the
	# robot.drive_wheels() function.
	# ####
	robot.drive_wheels(speed,speed,duration=dist/speed)

def my_turn_in_place(robot, angle, speed):
	"""Rotates the robot in place.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle -- Desired distance of the movement in degrees
		speed -- Desired speed of the movement in degrees per second
	"""
	# ####
	# TODO: Implement your version of a rotating in place function using the
	# robot.drive_wheels() function.
	# ####
	#when turning in place one of the wheels is center
	#so radius of circle is distance between wheels.
	#angle/2*pi = distance/2*pi*r
	#distance = r*angle = distancebetweenwheels*angle
	#speed = distance/time.
	#duration = distance/speed
	#positive angle = ccw, negative = cw
	#time has to be positive so we will account by the speed of wheel
	wheelspeed = math.radians(speed)*get_distance_between_wheels()/2
	if angle > 0:
		robot.drive_wheels(-wheelspeed,wheelspeed,duration=abs(angle/speed))
	else:
		robot.drive_wheels(wheelspeed,-wheelspeed,duration=abs(angle/speed))

def my_go_to_pose1(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	# ####
	# TODO: Implement a function that makes the robot move to a desired pose
	# using the my_drive_straight and my_turn_in_place functions. This should
	# include a sequence of turning in place, moving straight, and then turning
	# again at the target to get to the desired rotation (Approach 1).
	# ####

	dy = y - robot.pose.position.y
	dx = x - robot.pose.position.x
	rotation = math.degrees(math.atan2(dx,dy))
	finalRotation = angle_z-rotation#needs to be calculated now since when we move the angle changes
	my_turn_in_place(robot,rotation,25)
	my_drive_straight(robot,math.sqrt(x*x+y*y),25)
	my_turn_in_place(robot,finalRotation,25)


def normalize(angle):
    while angle < -math.pi:
		angle += 2 * math.pi
    while angle > math.pi:
		angle -= 2 * math.pi
    return angle

def my_go_to_pose2(robot, x, y, angle_z):

	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	# ####
	# TODO: Implement a function that makes the robot move to a desired pose
	# using the robot.drive_wheels() function to jointly move and rotate the 
	# robot to reduce distance between current and desired pose (Approach 2).
	# ####
	#get the goal in world coordinates so that when we move the goal isnt moving
	goalInWorld = cozmo.util.pose_z_angle(robot.pose.position.x + x, robot.pose.position.y + y, 0,cozmo.util.degrees(robot.pose.rotation.angle_z.degrees + angle_z))

	#formula combination taken from chapter 3.5
	while True:
		dy = goalInWorld.position.y - robot.pose.position.y
		dx = goalInWorld.position.x - robot.pose.position.x
		rho = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
		alpha = normalize( math.atan2(dy, dx) - robot.pose.rotation.angle_z.radians )
		eta = normalize(math.radians(goalInWorld.rotation.angle_z.degrees - robot.pose.rotation.angle_z.degrees))

		print("rho = {0}; alpha = {1}; eta = {2}".format(rho, alpha,eta))

		if rho < 10 :#if we are close and angle is small return
			if abs(math.degrees(eta)) < 5:
				robot.stop_all_motors()
				return
			else:#we are close so just mostly rotate
				p1=0.1
				p2 = 0.1
				p3=0.3
		else :#we are far and right direction so keep driving straight mostly
			if abs(math.degrees(alpha)) < 5:
				p1 = 0.5
				p2=.1
				p3=.1
			else :#far and not right direction do a combination of them
				p1 = 0.2
				p2=.2
				p3=.1

		xdot = min(p1*rho,25)
		thetadot = p2*alpha + p3*eta

		extraSpeed = thetadot * get_distance_between_wheels()/2
		rightspeed = xdot +  extraSpeed
		leftspeed = xdot - extraSpeed

		print("Left speed = {0}; Right speed = {1}".format(leftspeed, rightspeed))
		if abs(leftspeed) <3 and abs(rightspeed) <3:#when speed is this low the wheels tend not to turn
			leftspeed = leftspeed * 2
			rightspeed = rightspeed * 2
		robot.drive_wheels(leftspeed, rightspeed, duration = .5)


def my_go_to_pose3(robot, x, y, angle_z):
	"""Moves the robot to a pose relative to its current pose.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		x,y -- Desired position of the robot in millimeters
		angle_z -- Desired rotation of the robot around the vertical axis in degrees
	"""
	# ####
	# TODO: Implement a function that makes the robot move to a desired pose
	# as fast as possible. You can experiment with the built-in Cozmo function
	# (cozmo_go_to_pose() above) to understand its strategy and do the same.
	# ####
	#combo of 1 and 2. If the point is behind then do 1 (turn then go)
	angle = math.degrees(math.atan((y-robot.pose.position.y)/(x-robot.pose.position.x)))#get angle of the pose point relative to location
	if  abs(angle - robot.pose.rotation.angle_z.degrees) > 90:
		my_go_to_pose1(robot,x,y,angle_z)
	else :#if the point is within 45 degrees to either side of our center then do 2(curve to it)
		my_go_to_pose2(robot,x,y,angle_z)

def run(robot: cozmo.robot.Robot):

	print("***** Front wheel radius: " + str(get_front_wheel_radius()))
	print("***** Distance between wheels: " + str(get_distance_between_wheels()))

	## Example tests of the functions

	#cozmo_drive_straight(robot, 62, 50)
	#cozmo_turn_in_place(robot, 60, 30)
	#cozmo_go_to_pose(robot, 100, 100, 45)

	#rotate_front_wheel(robot, 90)
	#my_drive_straight(robot, 120, 60)
	#my_turn_in_place(robot, 90, 30)

	#my_go_to_pose1(robot, 100, 100, 45)
	#my_go_to_pose2(robot, 100, 100, 45)
	#my_go_to_pose3(robot, 100, 100, 45)


if __name__ == '__main__':

	cozmo.run_program(run)



