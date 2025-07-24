import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB','m11x11', 'm018PV010','m9x10','m9x16','m7x11','exp3']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def load_and_save2_general(category, filename, dataset, dataset_folder, shape):
	# 如果shape的tuple长度为1，则使用temp = np.zeros((shape[0], 1))；否则用temp = np.zeros(shape)
	temp = np.zeros(shape) if len(shape) > 1 else np.zeros((shape[0], 1))
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		# 如果shape的tuple长度为1，使用temp[start-1:end-1, :] = 1；否则使用temp[start-1:end-1, indx] = 1
		if len(shape) == 1:
			temp[start-1:end-1, :] = 1
		else:
			temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"{os.path.basename(dataset_folder)}/{dataset}_{category}.npy"), temp)

def load_and_save2_m11x11(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"m11x11/{dataset}_{category}.npy"), temp)

def load_and_save2_m018PV010(category, filename, dataset, dataset_folder, shape):
    # Initialize temp as a 1D array. 'shape' should be a single integer here.
    temp = np.zeros((shape[0], 1))

    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
       ls = f.readlines()

    for line in ls:
       pos, values = line.split(':')[0], line.split(':')[1].split(',')
       start, end = int(pos.split('-')[0]), int(pos.split('-')[1])
       temp[start-1:end-1, :] = 1

    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"m018PV010/{dataset}_{category}.npy"), temp)

def process_with_normalize_independent(filename, dataset_folder, output_folder):
	dataset_name = filename.strip('.txt')

	# Load data
	train_data = np.genfromtxt(os.path.join(dataset_folder, 'train', filename),
							   dtype=np.float64,
							   delimiter=',')
	test_data = np.genfromtxt(os.path.join(dataset_folder, 'test', filename),
							  dtype=np.float64,
							  delimiter=',')

	# Normalize each column independently
	train_data, _, _ = normalize2np_column(train_data)
	test_data, _, _ = normalize2np_column(test_data)

	# Save the processed data
	np.save(os.path.join(output_folder, f"{dataset_name}_train.npy"), train_data)
	np.save(os.path.join(output_folder, f"{dataset_name}_test.npy"), test_data)

	# Optional: Save normalization parameters
	# np.save(os.path.join(output_folder, f"SMD/{dataset_name}_min.npy"), min_a)
	# np.save(os.path.join(output_folder, f"SMD/{dataset_name}_max.npy"), max_a)

	# Process labels
	load_and_save2_general('labels', filename, dataset_name, dataset_folder, test_data.shape)

	print(f"Processed {dataset_name}: train {train_data.shape}, test {test_data.shape}")

def process_with_normalize2(filename, dataset_folder, output_folder):
	dataset_name = filename.strip('.txt')

	# 加载数据
	train_data = np.genfromtxt(os.path.join(dataset_folder, 'train', filename),
							   dtype=np.float64,
							   delimiter=',')
	test_data = np.genfromtxt(os.path.join(dataset_folder, 'test', filename),
							  dtype=np.float64,
							  delimiter=',')

	# 使用按列归一化
	train_data, min_a, max_a = normalize2np_column(train_data)
	test_data, _, _ = normalize2np_column(test_data, min_a, max_a)

	# 根据数据维度决定是否需要reshape
	if len(train_data.shape) == 1:
	    np.save(os.path.join(output_folder, f"{dataset_name}_train.npy"), train_data.reshape(-1, 1))
	    np.save(os.path.join(output_folder, f"{dataset_name}_test.npy"), test_data.reshape(-1, 1))
	else:
	    np.save(os.path.join(output_folder, f"{dataset_name}_train.npy"), train_data)
	    np.save(os.path.join(output_folder, f"{dataset_name}_test.npy"), test_data)

	# 可选：保存归一化参数
	# np.save(os.path.join(output_folder, f"SMD/{dataset_name}_min.npy"), min_a)
	# np.save(os.path.join(output_folder, f"SMD/{dataset_name}_max.npy"), max_a)

	# 处理标签
	load_and_save2_general('labels', filename, dataset_name, dataset_folder, test_data.shape)
	#load_and_save2_m11x11('labels', filename, dataset_name, dataset_folder, test_data.shape)
	#load_and_save2_m018PV010('labels', filename, dataset_name, dataset_folder, test_data.shape)

	print(f"Processed {dataset_name}: train {train_data.shape}, test {test_data.shape}")

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize2np_column(a, min_a = None, max_a = None):
    """按列进行归一化（每列单独计算最小值和最大值）"""
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 1e-8), min_a, max_a  # 添加小值避免除零

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'synthetic':
		train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
		test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
		dat = pd.read_csv(train_file, header=None)
		split = 10000
		train = normalize(dat.values[:, :split].reshape(split, -1))
		test = normalize(dat.values[:, split:].reshape(split, -1))
		lab = pd.read_csv(test_labels, header=None)
		lab[0] -= split
		labels = np.zeros(test.shape)
		for i in range(lab.shape[0]):
			point = lab.values[i][0]
			labels[point-30:point+30, lab.values[i][1:]] = 1
		test += labels * np.random.normal(0.75, 0.1, test.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'm11x11':
		dataset_folder = 'data/m11x11'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm018PV010':
		dataset_folder = 'data/m018PV010'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm9x10':
		dataset_folder = 'data/m9x10'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm9x16':
		dataset_folder = 'data/m9x16'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm7x11':
		dataset_folder = 'data/m7x11'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm7x12':
		dataset_folder = 'data/m7x12'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'exp3':
		dataset_folder = 'data/exp3'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm7x11_2024_1_9':
		dataset_folder = 'data/m7x11_2024_1_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm12x11_2023_7_9':
		dataset_folder = 'data/m12x11_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm15x11_2023_7_9':
		dataset_folder = 'data/m15x11_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm16x11_2023_7_9':
		dataset_folder = 'data/m16x11_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm21x11_2023_7_9':
		dataset_folder = 'data/m21x11_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm22x11_2023_7_9':
		dataset_folder = 'data/m22x11_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm12x16_2023_7_9':
		dataset_folder = 'data/m12x16_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm15x16_2023_7_9':
		dataset_folder = 'data/m15x16_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm16x16_2023_7_9':
		dataset_folder = 'data/m16x16_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm21x16_2023_7_9':
		dataset_folder = 'data/m21x16_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm22x16_2023_7_9':
		dataset_folder = 'data/m22x16_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm22x17_2023_7_9':
		dataset_folder = 'data/m22x17_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm12x17_2023_7_9':
		dataset_folder = 'data/m12x17_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm15x17_2023_7_9':
		dataset_folder = 'data/m15x17_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm16x17_2023_7_9':
		dataset_folder = 'data/m16x17_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'm21x17_2023_7_9':
		dataset_folder = 'data/m21x17_2023_7_9'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				process_with_normalize2(filename, dataset_folder, folder)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		dataset_folder = 'data/MSDS'
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
		_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
		train, _, _ = normalize3(df_train, min_a, max_a)
		test, _, _ = normalize3(df_test, min_a, max_a)
		labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		labels = labels.values[::1, 1:]
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI'
		ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		test['Time'] = test['Time'].astype(str)
		test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
		labels = test.copy(deep = True)
		for i in test.columns.tolist()[3:]: labels[i] = 0
		for i in ['Start Time', 'End Time']: 
			ls[i] = ls[i].astype(str)
			ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
		for index, row in ls.iterrows():
			to_match = row['Affected'].split(', ')
			matched = []
			for i in test.columns.tolist()[3:]:
				for tm in to_match:
					if tm in i: 
						matched.append(i); break			
			st, et = str(row['Start Time']), str(row['End Time'])
			labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
		train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")