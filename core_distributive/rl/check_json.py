import numpy as np
import json
from pprint import pprint
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH = 1

def __get__data(file_path=None, print_data=False):

	assert file_path != None

	data = json.load(open(file_path))
	print_data = {} # (change to dictionary)

	epi_success = []
	epi_loss = []
	curr_epoch = 0
	for episode,duration,episode_reward,loss,mae,mq,nbes,nb_steps,success_info in zip(data['episode'],data['duration'],data['episode_reward'],data['loss'],data['mean_absolute_error'],data['mean_q'],data['nb_episode_steps'],data['nb_steps'],data['infos']):

		#book keeping
		success_val = float(success_info.split(':')[1])
		epi_success.append(success_val)
		epi_loss.append(0 if math.isnan(loss) else loss)
		if(episode%EPOCH==0):
			if math.isnan(mq):
				mq=0
			success_val_mean = np.mean(epi_success)
			success_val_std = np.std(epi_success)
			loss_mean = np.mean(epi_loss)
			# can add more data if needed

			temp_data = [('epoch', curr_epoch), ('success_mean', success_val_mean), ('success_std', success_val_std), ('loss_mean', loss_mean)]
			for key, value in temp_data:
				if key not in print_data:
				    print_data[key] = []
				print_data[key].append(value)

			#reset
			epi_success = []
			epi_loss = []
			curr_epoch += 1
			#validate
			if(print_data==True):
				template = 'episode: {episode}, duration: {duration:.3f}s, episode_reward: {episode_reward:.3f}, loss: {loss:.3f}, mean_q: {mean_q:.3f}, success_rate: {success_info:.3f}'
				variables = {
			            'nb_steps': nb_steps,
			            'episode': episode + 1,
			            'duration': duration,
			            'mean_q': mq,
			            'episode_steps': nbes,
			            'sps': float(nbes) / duration,
			            'loss':loss,
			            'episode_reward': episode_reward,
			            'success_info':float(success_val)
			        }
				print(template.format(**variables))
	return print_data

def plot_af(file_path=None,save_file_name='temp_plot.png',plot_what='success'):

	if(file_path==None):
		print('could not find a path to training .json file')
	else:
		print_data = __get__data(file_path)	#print_data has form {'epoch': [], 'success_mean': [], 'success_std': []}
		# for (epoch,suc_mean,suc_std) in zip(print_data['epoch'],print_data['success_mean'],print_data['success_std']):
		extension = 'success_rate'
		if(plot_what=='success'):
			print("Plotting accuracy...")
			y = np.asarray(print_data['success_mean'])
			x = np.asarray(print_data['epoch'])
			e = np.asarray(print_data['success_std'])
			# plt.errorbar(np.squeeze(x), np.squeeze(y), np.squeeze(e), linestyle='None', marker='^')
			plt.plot(np.squeeze(x),np.squeeze(y), '-k')
			plt.fill_between(np.squeeze(x), np.squeeze(y) - np.squeeze(e), np.squeeze(y) + np.squeeze(e), color='blue', alpha=0.4)
		elif(plot_what=='loss'):
			print("Plotting loss...")
			y = np.asarray(print_data['loss_mean'])
			x = np.asarray(print_data['epoch'])
			plt.plot(np.squeeze(x), np.squeeze(y))
			extension = 'train_loss'
		plt.savefig(str(save_file_name)+'_'+extension+'.png', bbox_inches='tight')
		print("Plot saved.")
		# plt.show()

if __name__ == '__main__':
	plot_af('temp.json', plot_what='loss')
