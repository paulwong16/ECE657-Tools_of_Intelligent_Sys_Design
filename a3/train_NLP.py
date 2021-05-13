import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pickle import dump
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle


import utlis

if __name__ == "__main__":
	train_corpus, train_labels = utlis.get_data('./data/aclImdb/train/neg/*.txt', './data/aclImdb/train/pos/*.txt')
	train_corpus = utlis.normalize(train_corpus)
	# test_corpus, test_labels = utlis.get_data('./data/aclImdb/test/neg/*.txt', './data/aclImdb/test/pos/*.txt')
	# test_corpus = utlis.normalize(test_corpus)

	train_corpus, train_labels = shuffle(train_corpus, train_labels)
	# test_corpus, test_labels = shuffle(test_corpus, test_labels)

	tfidf_vectorizer, tfidf_train_features = utlis.tfidf_extractor(train_corpus)
	# tfidf_test_features = tfidf_vectorizer.transform(test_corpus)
	# count_vectorizer = CountVectorizer(min_df=0.1)
	# count_vectorizer_features = count_vectorizer.fit_transform(train_corpus)
	X = tfidf_train_features.toarray()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = utlis.MLP(X.shape[1])
	model = model.to(device)
	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []
	batch_size = 1024
	# n = X.shape[0]
	# split = int(n * 0.8)
	# X_train = X[:split]
	# X_val = X[split:]
	# y_train = train_labels[:split]
	# y_val = train_labels[split:]
	batch_num_train = X.shape[0] // batch_size + 1
	num_epochs = 7
	# batch_num_val = X_val.shape[0] // batch_size + 1
	#
	# for epoch in range(15):
	# 	model.train()
	# 	correct = 0
	# 	train_loss = 0
	# 	for batch_idx in range(batch_num_train):
	# 		start = batch_size * batch_idx
	# 		end = min(X_train.shape[0], batch_size * (batch_idx+1))
	# 		data = torch.from_numpy(X_train[start:end]).float().to(device)
	# 		target = torch.from_numpy(np.array(y_train[start:end])).long().to(device)
	# 		optimizer.zero_grad()
	# 		output = model(data)
	# 		pred = output.max(1, keepdim=True)[1]
	# 		correct += pred.eq(target.view_as(pred)).sum().item()
	# 		loss = criterion(output, target)
	# 		loss.backward()
	# 		optimizer.step()
	# 		train_loss += loss.item()
	# 	train_losses.append(train_loss)
	# 	accuracy = correct / X.shape[0]
	# 	train_accuracies.append(accuracy)
	# 	print('Train Epoch: %i, Loss: %f, Accuracy: %f' % (epoch+1, train_loss, accuracy))
	# 	model.eval()
	# 	correct = 0
	# 	test_loss = 0
	# 	with torch.no_grad():
	# 		for batch_idx in range(batch_num_val):
	# 			start = batch_size * batch_idx
	# 			end = min(X_val.shape[0], batch_size * (batch_idx + 1))
	# 			data = torch.from_numpy(X_val[start:end]).float().to(device)
	# 			target = torch.from_numpy(np.array(y_val[start:end])).long().to(device)
	# 			output = model(data)
	# 			pred = output.max(1, keepdim=True)[1]
	# 			correct += pred.eq(target.view_as(pred)).sum().item()
	# 			loss = criterion(output, target)
	# 			test_loss += loss.item()
	# 	accuracy = correct / X_val.shape[0]
	# 	val_losses.append(test_loss)
	# 	val_accuracies.append(accuracy)
	# 	print('Val Epoch:   %i, Loss: %f, Accuracy: %f' % (epoch+1, test_loss, accuracy))
	#
	# plt.figure()
	# plt.title('Train & validation loss')
	# plt.title('Train loss')
	# plt.plot([i + 1 for i in range(30)], train_losses)
	# plt.plot([i + 1 for i in range(30)], val_losses)
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend(['Train', 'Validation'])
	# plt.show()
	# plt.figure()
	# plt.title('Train & validation accuracy')
	# plt.plot([i + 1 for i in range(30)], train_accuracies)
	# plt.plot([i + 1 for i in range(30)], val_accuracies)
	# plt.xlabel('Epoch')
	# plt.ylabel('Accuracy')
	# plt.legend(['Train', 'Validation'])
	# plt.show()

	for epoch in range(num_epochs):
		model.train()
		correct = 0
		train_loss = 0
		for batch_idx in range(batch_num_train):
			start = batch_size * batch_idx
			end = min(X.shape[0], batch_size * (batch_idx+1))
			data = torch.from_numpy(X[start:end]).float().to(device)
			target = torch.from_numpy(np.array(train_labels[start:end])).long().to(device)
			optimizer.zero_grad()
			output = model(data)
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(target.view_as(pred)).sum().item()
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_losses.append(train_loss)
		accuracy = correct / X.shape[0]
		train_accuracies.append(accuracy)
		print('Train Epoch: %i, Loss: %f, Accuracy: %f' % (epoch+1, train_loss, accuracy))

	plt.figure()
	plt.title('Train loss')
	plt.plot([i + 1 for i in range(num_epochs)], train_losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	plt.figure()
	plt.title('Train Accuracy')
	plt.plot([i + 1 for i in range(num_epochs)], train_accuracies)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.show()

	save_model = {'tf_vector': tfidf_vectorizer, 'model': model}

	dump(save_model, open('./models/20856733_NLP_model.pkl', 'wb'))

	pass