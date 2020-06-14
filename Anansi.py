# Real time audio tester. Tests against the list of phrase signatures

import os
import json
import pickle
import Sampler
import numpy as np

class Anansi(object):
	accuracy = None
	sigs = None
	realAudio = None
	rmsAccuracy = None

	# ----------------------------------------------------------------------

	def __init__(self):

		with open('Global_Variables.json') as var:
			args = json.load(var)

		self.accuracy = args['accuracy']
		self.rmsAccuracy = args['rmsAccuracy']

		path = os.getcwd() + '\\Brine\\Signatures.pickle'

		pickle_in = open(path, 'rb')
		self.sigs = pickle.load(pickle_in)
		pickle_in.close()

		# Save the last two seconds or so of audio data
		# Initialize as zeros
		self.realAudio = np.zeros(list(self.sigs.values())[0].shape)

	# ----------------------------------------------------------------------

	def callback(self, realTimeFFT, rmsMethod = False):
		self.realAudio = np.concatenate((self.realAudio[1:], [realTimeFFT]))

		for phrase, sig in self.sigs.items():

			if rmsMethod:
				diff = Sampler.rms(self.realAudio, sig)

				if diff < self.rmsAccuracy:
					print(phrase)
					# do something
			else:
				diff = Sampler.quantify_diff(self.realAudio, sig)				

				if diff > self.accuracy:
					print(phrase)
					# do something




				if phrase == 'Down' and diff > 0.963:
					print(diff)
				if phrase == 'Down' and diff < 1.1:
					print(diff)