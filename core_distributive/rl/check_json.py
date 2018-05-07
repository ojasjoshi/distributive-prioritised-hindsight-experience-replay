import numpy as np
import json
from pprint import pprint
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_json import process_json

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

def plot_af(file_path=None,preprocess_json=preprocess_json,save_file_name='temp_plot.png',plot_what='success',plot_num=0,label=None,color_vector=['blue', 'red', 'black', 'green', 'magenta', 'green','cyan', 'yellow', 'brown','gray','olive','orange','pink','salmon','hotpink','palegoldenrod','mediumseagreen','sienna','tomato']):

	if(file_path==None):
		print('could not find a path to training .json file')

	else:
		if(preprocess_json):
			process_json(file_path)
			file_path = "data/temp.json"
		print_data = __get__data(file_path)	#print_data has form {'epoch': [], 'success_mean': [], 'success_std': []}
		extension = 'success_rate'
		if(plot_what=='success'):
			print("Plotting accuracy...")
			y = np.asarray(print_data['success_mean'])
			x = np.asarray(print_data['epoch'])
			e = np.asarray(print_data['success_std'])
			if(e.shape[0]>1):
				plot, = plt.plot(np.squeeze(x),np.squeeze(y),label=label,linewidth=1.0,color=color_vector[plot_num])
				plt.fill_between(np.squeeze(x), np.squeeze(y)-np.squeeze(e), np.squeeze(y)+np.squeeze(e), color=color_vector[plot_num], alpha=0.2, label=label)
			else:
				plot = None
			
		elif(plot_what=='loss'):
			print("Plotting loss...")
			y = np.asarray(print_data['loss_mean'])
			x = np.asarray(print_data['epoch'])
			plt.plot(np.squeeze(x), np.squeeze(y))
			extension = 'train_loss'
		print("Plot saved.")
		return plot

if __name__ == '__main__':
	if(len(sys.argv)==2):
		path = str(sys.argv[1])
	##graph0	
	# label_names = ['DDPG', 'PER']
	##graph1
	# label_names = ['Vanilla DDPG','"episode" strategy with K=1', '"episode" strategy with K=4', '"episode" strategy with K=8', '"future" strategy with K=1', '"future" strategy with K=4', '"future" strategy with K=8']
	##graph 2
	# label_names = ['"episode" strategy with K=4', '"episode" strategy with K=8', '"episode" strategy with K=24', '"future" strategy with K=4', '"future" strategy with K=8', '"future" strategy with K=24']
	##graph3_1
	# label_names = ['"episode" strategy with alpha=0.3', '"episode" strategy with alpha=0.7', '"episode" strategy with alpha=0.9', '"episode" strategy with alpha=0']
	##graph3_2
	# label_names = ['"future" strategy with alpha=0.3', '"future" strategy with alpha=0.7', '"future" strategy with alpha=0.9', '"future" strategy with alpha=0']
	##graph4
	# label_names = ['"episode" strategy with K=4', '"episode" strategy with K=8', '"future" strategy with K=4', '"future" strategy with K=8']
	##graph5
	# label_names = ['Unprioritized Actors', '1 Actor Exploring - 1 Actor Exploiting', 'Both Actors Exploiting', 'Dynamic Actors']
	##graph6
	# label_names = ['Memory= 10000, Batch Size = 2', 'Memory= 10000, Batch Size = 8', 'Memory= 50000, Batch Size = 2', 'Memory= 50000, Batch Size = 8', 'Memory= 50000, Batch Size = 64']
	##graph8
	label_names = ['Vanilla-DDPG','DDPG+PER','DDPG+HER','DDPG+PHER','DDPG+DPHER']
	# graph_12
	# label_names = ['Memory= 5000, Batch Size = 32', 'Memory= 10000, Batch Size = 2', 'Memory= 10000, Batch Size = 8', 'Memory= 50000, Batch Size = 2', 'Memory= 50000, Batch Size = 8', 'Memory= 50000, Batch Size = 64']

	plt.clf()
	plot_handles = []
	f = open(path,'r')
	for i, (exp_name,label_name) in enumerate(zip(f.readlines(), label_names)):
		if(exp_name!="\n"):
			exp_path = os.path.join("data", exp_name)
			plot = plot_af(exp_path[:-1], plot_what='success', plot_num=i,label=label_name)
			if plot!=None:
				plot_handles.append(plot)

	plt.xlabel('EPOCHS',size=14)
	plt.ylabel('Accuracy',size=14)
	plt.legend(handles=plot_handles, loc=4, prop={'size': 8})
	plt.savefig(path[:-4]+'.png', bbox_inches='tight', dpi=300)
