# Import necessary libraries
import torch
import sounddevice as sd
from glob import glob
from scipy.io.wavfile import write
import rospy
from std_msgs.msg import String

if __name__ == "__main__":
	rospy.init_node('stt')
	# Load the model
	device = torch.device('cpu') # gpu also works, but the models are fast enough for CPU
	model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language='en', device=device, verbose=False) # 'de' and 'es' languages are also available
	(read_batch, split_into_batches, read_audio, prepare_model_input) = utils

	pub = rospy.Publisher('command', String, queue_size=10)
	while not rospy.is_shutdown():
		print('Say something nice')
		# Record audio from the microphone
		duration = 5  # seconds
		fs = 16000
		recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
		sd.wait()
		write('output.wav', fs, recording)  # Save as WAV file

		# Prepare batches
		test_files = glob('output.wav')
		batches = split_into_batches(test_files, batch_size=10)

		# Prepare model input
		input = prepare_model_input(read_batch(batches[0]), device=device)

		# Get model output
		output = model(input)
		# Decode the output
		msg = String()
		for example in output:
			msg.data = decoder(example.cpu())
			
		rospy.loginfo(msg)
		pub.publish(msg)
		rospy.sleep(1)