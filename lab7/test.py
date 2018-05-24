#!/usr/bin/env python3

import cozmo
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps
import math
import time
import  odometry

def run(robot: cozmo.robot.Robot):
	#odometry.cozmo_drive_straight(robot,87,25)
	#robot.drive_wheels(100,60,duration=14.2)
	odometry.rotate_front_wheel(robot,360)
	return

if __name__ == '__main__':

	cozmo.run_program(run)


