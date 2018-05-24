#!/usr/bin/env python3

import asyncio
import sys

import cv2
import numpy as np

import imgclassification
import math
import cozmo
from collections import Counter

def main(robot: cozmo.robot.Robot):

	global classifier


	try:

		predictions = Counter()#structure to keep count per entry
		sum = 0#sum of all enteries in structure
		while True:

			robot.set_head_angle(cozmo.util.degrees(-5)).wait_for_completed()#i found this angle optimal for viewing

			robot.camera.image_stream_enabled = True
			# get camera image
			event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

			# convert camera image to opencv format
			opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)

			#predict image
			img_features = classifier.extract_image_features([opencv_image])
			label = str(classifier.predict_labels(img_features)[0])
			print(label)
			#tract prediction ratio
			if label == "none":
				continue
			predictions[label] = predictions[label]+1
			sum = sum + 1

			# do an animation if the label has been see atleast 10 times and is more than 50% of labels
			if predictions[label]>10 and (predictions[label]/sum)>.5:
				if label == 'drone':
					robot.say_text("Thats a drone").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.DroneModeCliffEvent).wait_for_completed()
				elif label == 'inspection':
					robot.say_text("Thats an inspection").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.HiccupSelfCure).wait_for_completed()
				elif label == 'order':
					robot.say_text("Thats an order").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.CozmoSaysSpeakGetOutShort).wait_for_completed()
				elif label == 'plane':
					robot.say_text("Thats a plane").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.PetDetectionShort_Cat).wait_for_completed()
				elif label == 'truck':
					robot.say_text("Thats a truck").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabFireTruck).wait_for_completed()
				elif label == 'hands':
					robot.say_text("Thats a hand").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.MemoryMatchPlayerWinHandSolo).wait_for_completed()
				elif label == 'place':
					robot.say_text("Thats a place").wait_for_completed()
					robot.play_anim_trigger(cozmo.anim.Triggers.GuardDogPlayerSuccess).wait_for_completed()

				#clear the list once we have seen and responded to an image
				predictions.clear()
				sum = 0

	except KeyboardInterrupt:
		print("")
		print("Exit requested by user")
	except cozmo.RobotBusy as e:
		print(e)


if __name__ == "__main__":
	# initialize and train the classifier
	classifier = imgclassification.ImageClassifier()
	(train_raw, train_labels) = classifier.load_data_from_folder('./train/')
	train_data = classifier.extract_image_features(train_raw)
	classifier.train_classifier(train_data, train_labels)
	print("done training")
	cozmo.run_program(main, force_viewer_on_top = True)