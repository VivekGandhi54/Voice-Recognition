# Signatures averages all the spectrograms to find what a typical
# spectrogram for the phrase should look like
#
# An issue we have with multiple samples is that their whitespacing differs.
# One clip may start at second 0.4, and another at 1.2. So, we move one spectrogram
# over the other, 50 ms <block duration> at a time. When they correlate the best,
# we take their average

import numpy as np
import Sampler
import pickle
import json
import os

# ==========================================================================
# Load relevant global variables

with open('Global_Variables.json') as var:
	args = json.load(var)

phi = args['Phi']

# ==========================================================================

def generateSignatures(rmsMethod = False):
	allSampleSpecs = Sampler.unpickleRawSpectrograms()
	signatures = {}

	for phrase, allSamples in allSampleSpecs.items():

		# Find the largest spectrogram in the list. Frame the signature to fit this
		maxSize = max([each.shape[0] for each in allSamples])
		largest = list(filter(lambda x: x.shape[0] == maxSize, allSamples))[0]

		largest_norm = np.copy(largest)
		
		if not rmsMethod:
			largest_norm[largest_norm < phi] = 0
			largest_norm[largest_norm != 0] = 1

		# Add closest copies to base to average out the spectrograms
		base = np.copy(largest)

		# ------------------------------------------------------------------

		for other in allSamples:
			if np.array_equal(other, largest):
				continue

			other_norm = np.copy(other)

			if not rmsMethod:
				other_norm[other_norm < phi] = 0
				other_norm[other_norm != 0] = 1

			minVal = 1000
			minIndex = 0

			maxVal = 0
			maxIndex = 0

			a = largest.shape[0]
			b = other.shape[0]

			# --------------------------------------------------------------
			# I will shoot you if you ask me to redo or explain this math
			# x is the length of upper zero padding
			# z is the length of lower zero padding
			# y is the length of the moving array
			# U and L are upper and lower indices

			def indices(i):
				x = a - 1 - i
				x = 0 if x < 0 else x

				z = i - b + 1
				z = 0 if z < 0 else z

				y = a - x - z

				U = 0
				U = i - a + 1 if i >= a else U
				L = U + y

				return {'x': x,
						'z': z,
						'U': U,
						'L': L,}

			# --------------------------------------------------------------
			# Shift the selected normalized spectrogram across the largest, one pixel at a time,
			# and record when they match the best

			for i in range(a + b - 1):
				ind = indices(i)
				x = ind['x']
				z = ind['z']
				U = ind['U']
				L = ind['L']

				XArray = np.zeros((x, largest.shape[1]))
				YArray = other_norm[U:L]
				ZArray = np.zeros((z, largest.shape[1]))
				
				# 1 pixel moved array to compare to
				compare = np.concatenate((XArray, YArray, ZArray))

				if rmsMethod:
					diff = Sampler.rms(compare, largest_norm)
				else:
					diff = Sampler.quantify_diff(compare, largest_norm)

				if diff < minVal:
					minVal = diff
					minIndex = i

				if diff > maxVal:
					maxVal = diff
					maxIndex = i

			# --------------------------------------------------------------

			# Recreate the matched spectrogram, but with non-normalized data

			if rmsMethod:
				ind = indices(minIndex)
			else:
				ind = indices(maxIndex)

			x = ind['x']
			z = ind['z']
			U = ind['U']
			L = ind['L']

			XArray = np.zeros((x, largest.shape[1]))
			YArray = other[U:L]
			ZArray = np.zeros((z, largest.shape[1]))

			appendSpec = np.concatenate((XArray, YArray, ZArray))

			# Add all the non-normalized data to average them, and then
			# normalize them again
			base = np.add(appendSpec, base)

		# ------------------------------------------------------------------

		# We've just been adding to the same spectrogram
		# Now, we take their average by dividing by number of samples
		base = base/len(allSamples)

		# Normalize it to save it as a signature
		if not rmsMethod:
			base[base < phi] = 0
			base[base != 0] = 1

		signatures[phrase] = base

		# ------------------------------------------------------------------

	# Add zeros so that they're all the same size. Useful for comparing
	
	# maxSize = max([each.shape[0] for each in signatures.values()])

	# for phrase, sig in signatures.items():
	# 	signatures[phrase] = np.concatenate((np.zeros((maxSize - sig.shape[0], sig.shape[1])), sig))

	# ----------------------------------------------------------------------
	# Remove lines until they're all the same size. Useful for comparing

	minSize = min([each.shape[0] for each in signatures.values()])

	for phrase, sig in signatures.items():

		while sig.shape[0] > minSize:
			if sum(sig[0]) > sum(sig[-1]):
				sig = sig[:-1]
			else:
				sig = sig[1:]

		signatures[phrase] = sig

	# ----------------------------------------------------------------------

	# Take a dump in a jar of brine
	pickle_out = open(os.getcwd() + '\\Brine\\Signatures.pickle', 'wb')
	pickle.dump(signatures, pickle_out)
	pickle_out.close()

	# Saved to Signatures.pickle as:
	# {	'Down': <signature>,
	#	'Up': <signature>, ... }

# ==========================================================================

generateSignatures(True)
