# Create a live audio input stream, then creates the running absolute fft
from pydub import AudioSegment as AS
from Anansi import Anansi
import sounddevice as sd
import numpy as np
import json
import os

# ==========================================================================
# block_duration:	Milliseconds
# fftsize:			Length of the fft = fftsize/2 + 1
# samplerate:		Default device samplerate is usually 44100 samples/second

def audioInputStream(rmsMethod = False):
	# Load relevant global variables

	with open('Global_Variables.json') as var:
		args = json.load(var)

	block_duration = args['block_duration']
	fftsize = args['n-point']
	samplerate = args['samplerate']
	phi = args['Phi']

	# ----------------------------------------------------------------------

	runner = Anansi()

	try:
		if samplerate == -1:
			samplerate = sd.query_devices(None, 'input')['default_samplerate']

		#-------------------------------------------------------------------		
		# Internal helper function to compute the fft
		# magnitude is an array (numpy.ndarray) that consists of (numpy.float64)
		# Audio data (indata) in the float range [-1, 1]
		# Audio block sample size: samplerate * block_duration / 1000 [default = 2250]

		def callback(indata, frames, time, status):
			dft = np.abs(np.fft.rfft(indata[:, 0], n = fftsize))

			if not rmsMethod:
				dft[dft < phi] = 0
				dft[dft != 0] = 1

			runner.callback(dft, rmsMethod)

			# print(max(dft))

			# do stuff

		#-------------------------------------------------------------------		

		with sd.InputStream(device = None, channels = 1, callback = callback,
							blocksize = int(samplerate * block_duration / 1000),
							samplerate = samplerate):
			while True:
				input()	# Press any key to exit
				break

	except KeyboardInterrupt:
		print('Interrupted by user')
	except Exception as e:
		print(e)
	finally:
		print('Terminated')

# ==========================================================================
# Reads audio file of any format
# Returns the sampling rate Fs and an numpy array of the signal data

def read(file, normalized = True):
	audio = AS.from_file(file, file.split('.')[-1])
	data = np.array(audio.get_array_of_samples())

	if audio.channels == 2:
		data = data.reshape((-1, 2))		# Stacked to side-by-side channels
		data = (data[:,0] + data[:,1])/2	# Average the channels: stereo to mono

	# Change limits from [0, 255] to [-1, 1]
	if normalized:
		return audio.frame_rate, np.float32(data) / 2**15

	return audio.frame_rate, data

# ==========================================================================
# Find list of all sample paths in 'Samples' directory
#
# sampleDict = {'Down': [<path 1>, <path 2>, ..., <path n>],
#				'Fire': [<path 1>, <path 2>, ..., <path n>],
#				<Phrase>: <paths>, ... }

def findSamples():
	samplesDir = os.getcwd() + '\\Samples'
	phrases = os.listdir(samplesDir)		# List of phrases ['a', 'b', 'c']

	sampleDict = {}

	for each in phrases:
		phraseDir = samplesDir + '\\' + each
		recordingNames = os.listdir(phraseDir)
		sampleDict[each] = [phraseDir + '\\' + each for each in recordingNames]

	return sampleDict

# ==========================================================================
