from matplotlib import pyplot as plt
from matplotlib import cm
import sounddevice as sd
import AudioInput as AI
import numpy as np
import pickle
import json
import os

# ==========================================================================
# Load relevant global variables

with open('Global_Variables.json') as var:
	args = json.load(var)

block_duration = args['block_duration']
fftsize = args['n-point']
phrases = args['phrases']
phi = args['Phi']

# ==========================================================================
# Returns the spectrogram for the sample at samplePath
# Not normalized yet
#
# Input: String path to the sample
#	"C:\\Users\\vgand\\Desktop\\Code\\Project\\Samples\\Fire\\Recording - Keziah May.m4a"
#
# Example output: signature is a numpy array
# signature =     [ [4.04, 9.56, 4.35 ... 6.76, 6.45, 9.15]
#					[1.29, 3.70, 2.36 ... 8.10, 1.42, 9.15]
#					[3.93, 2.11, 1.26 ... 1.73, 1.16, 6.10]
#					...
#					[9.08, 3.75, 3.86 ... 1.95, 6.41, 1.03]
#					[7.95, 2.68, 1.69 ... 6.69, 2.80, 5.41]
#					[8.77, 5.63, 4.70 ... 2.67, 1.63, 1.22] ]
#
# Newest data at bottom. Left to right = DFT of 1 to N/2 - 1

def rawSpectrogram(samplePath):
	fs, sample = AI.read(samplePath)

	block_size = int(fs * block_duration / 1000)	# Chunk of audio in each slice

	cols = fftsize//2 + 1		# Pre-calculate dimensions to make things easier
	rows = int(np.floor(len(sample)/block_size))

	signature = np.zeros((rows, cols))				# Initialize empty array

	for i in range(rows):
		arr = np.array_split(sample, [block_size])
		section = arr[0]		# Crop out the first block of the data chunk
		sample = arr[1]			# Save the rest for the next iteration

		dft = np.abs(np.fft.rfft(section, n = fftsize))

		signature[i] = dft		# Append dft to bottom of spectrogram

	return signature

# ==========================================================================
# Find the difference between two normalized spectrograms
# If the values at the same index are the same, place a "True" placeholder ther
# Return total count(True)/count(All) decimal between 0 and 1

def quantify_diff(spec1, spec2):
	assert (spec1.shape == spec2.shape), 'Spectrograms not of same dimensions'

	unique, counts = np.unique((spec1 == spec2), return_counts = True)
	diff = dict(zip(unique, counts))
	# Example diff = {False: 1231, True: 5090}

	# spec1 == spec2 is an array of booleans
	# Useful for printing the difference
	# To get a 1/0 equality array, use this:

	# 	diff = np.array(list(map((lambda x, y: x - y + 1), spec1, spec2)))
	# 	diff[diff != 1] = 0
	# 	printSpectrogram(diff)

	if True not in diff.keys():
		return 0
	if False not in diff.keys():
		return 1

	return diff[True]/(diff[True] + diff[False])

# ==========================================================================
# Prints out a spectrogram, normalized or not
# NOT to be used during runtime, purely for testing
# Only works well in Windows Terminal. Not cmd, PowerShell, etc.
# Uses ANSI escape codes

def printSpectrogram(spectrogram):
	# spectrogram = spectrogram/np.max(spectrogram)

	colors = 30, 34, 35, 91, 93, 97
	chars = ' :%#\t#%:'
	# chars = '        '
	gradient = []

	# Create a gradient of colors based on the colors and chars
	for bg, fg in zip(colors, colors[1:]):
		for char in chars:
			if char == '\t':
				bg, fg = fg, bg
			else:
				gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))

	low_bin = 1						# Frequency lower limit
	columns = spectrogram.shape[1]	# Frequency upper limit

	# UN/COMMENT THIS to see the gradient
	# print('\n  Gradient: |' + ' '.join(gradient) + '\x1b[0m')
	# print('  Values:   |' + '    0.'.join(map(str,list(range(10)))) + '    1')

	# For each row of the spec
	# clip() truncates values under 0 and above 1 to 0 and 1 respectively
	for magnitude in spectrogram:
		line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
				for x in magnitude[low_bin:low_bin + columns])

		print(*line, sep = '', end = '\x1b[0m\n')

	print('-'*128 + '\n')

# ==========================================================================
# Generate and pickle all the sample spectrograms. Not normalized by default
# Saves to cwd/Brine
# Saves as cwd/Brine/Up.pickle
#          cwd/Brine/Down.pickle
#          ...
# Multiple dumps in same pickle

def generateBrine(normalize = False):
	sampleDict = AI.findSamples()

	for phrase in sampleDict:
		pickle_out = open(os.getcwd() + '\\Brine\\' + phrase + '.pickle', 'wb')

		for path in sampleDict[phrase]:
			spec = rawSpectrogram(path)
			
			if normalize:
				spec[spec < phi] = 0
				spec[spec != 0] = 1

			pickle.dump(spec, pickle_out)

		pickle_out.close()

# ==========================================================================
# Deletes all old Pickled data. Run this when testing with new parameters

def cleanBrine():
	path = os.getcwd() + '\\Brine\\'

	failed = False
	for each in os.listdir(path):
		try:
			os.remove(path + each)
		except:
			failed = True

	if failed:
		print('Could not clean Brine, check manually')

# ==========================================================================
# Returns a dictionary with previously pickled spectrograms:
# output = { 'Up': [<spec1>, <spec2>, ... ],
#			 'Down': [<spec1>, <spec2>, ... ], ... }
# By loading in using arg phrases, we don't accidentally load Signatures

def unpickleRawSpectrograms():
	path = os.getcwd() + '\\Brine\\'

	if os.listdir(path) == []:
		raise Exception('No pickles in Brine')

	output = {}

	for each in phrases:
		pickle_in = open(path + each + '.pickle', 'rb')
		spectrograms = []

		try:
			while True:
				spectrograms.append(pickle.load(pickle_in))
		except:
			pass	# When we've loaded everything

		pickle_in.close()
		output[each] = spectrograms

	return output

# ==========================================================================
# Returns the root mean squares of two signals

def rms(a, b):
	assert (a.shape == b.shape), 'Arrays not of same size'

	a = a.flatten()
	b = b.flatten()
	
	return (sum((a - b)**2)/len(a))**0.5

# ==========================================================================

def printSpectrogram2(spec, phrase = 'Spectrogram'):

	fs = sd.query_devices(None, 'input')['default_samplerate']

	xlist = np.linspace(0, spec.shape[0]*block_duration, spec.shape[0])
	ylist = np.linspace(0, fs/2000, spec.shape[1])
	X, Y = np.meshgrid(xlist, ylist)

	spec = np.transpose(20*np.log10(spec))
	# spec = np.transpose(spec)

	fig,ax=plt.subplots(1,1)
	cp = ax.contourf(X, Y, spec, cmap = cm.YlGnBu) #BuPu
	fig.colorbar(cp)

	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Frequency (kHz)')
	# ax.set_title(title)

	# plt.savefig(phrase + '.jpg', bbox_inches='tight')
	plt.show()

# ==========================================================================

def filter(listOfWs, spectrogram = None):
	omegaList = np.arange(0, np.pi, 2*np.pi/fftsize)
	omegaList = np.append(omegaList, np.pi)

	y = np.ones(omegaList.size)

	for w in listOfWs:
		R = 0.97 # Cos why not

		z1 = R*np.exp(-1j*w)
		z2 = R*np.exp(1j*w)

		a1 = z1 + z2
		a2 = z1*z2

		H = lambda omega: 1/(1 - a1*np.exp(-1j*omega) + a2*np.exp(-2j*omega ))
	
		y = y * np.abs(np.array(list(map(H, omegaList))))

	other = np.array([each*y for each in spectrogram])

	return other, y

# ==========================================================================
