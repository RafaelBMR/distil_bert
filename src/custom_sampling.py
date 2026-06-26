import math
import logging
import argparse

from collections import Counter

from torch.utils.data import DataLoader, ConcatDataset, Subset


logger = logging.getLogger(__name__)


def apply_oversampling(dataloader: DataLoader, args: argparse.Namespace) -> DataLoader:
	# We copy the dataset from the original DataLoader and create a new
	# DataLoader with oversampled samples
	original_dataset = dataloader.dataset
	# Deciding the samples that will be oversampled, and how many times
	# We use a hash function to map all samples of each class for fast retrieval later
	samples_idx_per_label = {}
	label_counter = Counter()
	for idx, elem in enumerate(original_dataset):
		label_counter.update(elem['labels'])
		for label in elem['labels']:
			try:
				samples_idx_per_label[label].append(idx)
			except KeyError:
				samples_idx_per_label[label] = [idx]

	# Computing per class ratio
	per_class_ratio = {}
	for label, samples in samples_idx_per_label.items():
		n_samples = len(samples)
		# Classes containing the target amount are skiped,
		# because they already have enough samples
		if n_samples >= args.oversampling_target:
			continue
		class_ratio = int(math.sqrt(args.oversampling_target/n_samples))
		per_class_ratio[label] = class_ratio
	logger.info("Class ratios for oversampling: {}".format(per_class_ratio))

	# Determining the number of copies for each sample
	samples_copies = {}
	for idx, elem in enumerate(original_dataset):
		sample_ratio = 0
		for label in elem['labels']:
			if label in per_class_ratio.keys():
				sample_ratio = max(sample_ratio, per_class_ratio[label])
		# Cap by maximum
		sample_ratio = min(sample_ratio, args.oversampling_max_copies)
		if sample_ratio > 0:

			samples_copies[idx] = sample_ratio

	# Creating extra datasets
	extra_datasets = []
	for n_copies in range(1, args.oversampling_max_copies+1):
		indices = []
		for sample_idx, sample_ratio in samples_copies.items():
			if sample_ratio <= n_copies:
				indices.append(sample_idx)
		extra_datasets.append(Subset(original_dataset, indices))

	logger.info("Extra datasets sizes: {}".format([len(d) for d in extra_datasets]))

	# Generating new DataLoader
	new_dataset = ConcatDataset([original_dataset] + extra_datasets)

	new_dataloader = DataLoader(dataset=new_dataset,
							    batch_size=args.batch_size,
							    shuffle=True,
							    collate_fn=dataloader.collate_fn)
	return new_dataloader
