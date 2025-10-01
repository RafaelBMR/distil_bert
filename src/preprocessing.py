import os
import json

import pandas as pd

import torch

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from transformers import DistilBertTokenizerFast




def load_data(args):
	# Loading train, validation and test files
	with open(os.path.join(args.dataset_path, "train.json")) as f:
		train_data = json.loads(f.read())
	with open(os.path.join(args.dataset_path, "validation.json")) as f:
		validation_data = json.loads(f.read())
	with open(os.path.join(args.dataset_path, "test.json")) as f:
		test_data = json.loads(f.read())

	# Loading class mapping and converting ids to integers
	with open(os.path.join(args.dataset_path, "id2label.json")) as f:
		id2label = json.loads(f.read())
	id2label = {int(k): v for k, v in id2label.items()}

	data = {
		"train": train_data,
		"validation": validation_data,
		"test": test_data,
	}


	return (data, id2label)


def prepare_data_subset(data, tokenizer, shuffle, num_classes, args):
	"""
		Process a subset of the data (train, validation or test split).
		Texts are truncated to the max length.

		"data" argument must be a list of dicts, wehere each dict has a 'text'
		and a 'label' attributes.

		Returns a DataLoader ready for mini-batch training/evaluation.
	"""

	# Getting a list of texts for tokenizing later, while setting a binary
	# tensor indicating the labels of each sample
	texts = []
	labels = torch.zeros((len(data), num_classes))
	for sample_idx, sample in enumerate(data):
		texts.append(sample['text'])
		for label_id in sample['labels']:
			labels[sample_idx][label_id] = 1

	# Encoding the input text	
	encoded_texts = tokenizer.batch_encode_plus(texts,
												truncation=True,
												max_length=args.max_length)


	# Preparing the DataLoader with dynamic padding
	dataset = [
		{
			"input_ids": torch.tensor(encoded_texts['input_ids'][i]),
			"attention_mask": torch.tensor(encoded_texts['attention_mask'][i]),
			"labels": labels[i]
		}
		for i in range(len(data))
	]

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	data_loader = DataLoader(dataset=dataset,
							 batch_size=args.batch_size,
							 shuffle=shuffle,
							 collate_fn=data_collator)

	return data_loader






def prepare_data(data, num_classes, args):
	"""
		Converts the dataset into DistilBert's format and make it available
		as dataloaders. 
		Training data is set to shuffle, and dataloaders provide dynamic padding.
	"""

	# Loading the tokenizer
	tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name_or_path)

	# Converting input data to dataloaders
	train_dataloader = prepare_data_subset(data['train'], tokenizer, shuffle=True, 
										   num_classes=num_classes, args=args)
	validation_dataloader = prepare_data_subset(data['validation'], tokenizer, shuffle=False, 
										        num_classes=num_classes, args=args)
	test_dataloader = prepare_data_subset(data['test'], tokenizer, shuffle=False, 
										  num_classes=num_classes, args=args)

	dataloaders = {
		"train": train_dataloader,
		"validation": validation_dataloader,
		"test": test_dataloader
	}

	return dataloaders
	
