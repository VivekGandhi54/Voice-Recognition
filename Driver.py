import AudioInput as AI
import numpy as np
import Sampler
import Signatures
import pickle, json, os

# ==========================================================================
# Load relevant global variables

with open('Global_Variables.json') as var:
	args = json.load(var)

block_duration = args['block_duration']
fftsize = args['n-point']
phi = args['Phi']
phrases = args['phrases']

# ==========================================================================

# Generate signatures
# Sampler.cleanBrine()
# Sampler.generateBrine()
# Signatures.generateSignatures(True)

# Run real-time test
# AI.audioInputStream(True)

# Signatures.generateSignatures(True)


path = os.getcwd() + '\\Brine\\Signatures.pickle'
pickle_in = open(path, 'rb')
sigs = pickle.load(pickle_in)
pickle_in.close()


# for phrase, sig in sigs.items():

# print(phrases)

# Sampler.printSpectrogram2(sigs[phrases[1]])

Sampler.printSpectrogram2(sigs[phrases[4]])
