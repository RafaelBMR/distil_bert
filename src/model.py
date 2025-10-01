import os
import shutil

from tqdm import tqdm

import torch
from torch.optim import AdamW

from transformers import DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

import mlflow
import mlflow.pytorch

from torch.utils.tensorboard import SummaryWriter

from log_utils import LOG_LAYERS



def load_model(name_or_path, id2label):

	model = DistilBertForSequenceClassification.from_pretrained(
				pretrained_model_name_or_path=name_or_path,
				problem_type="multi_label_classification",
				id2label=id2label,
				label2id={v: k for k, v in id2label.items()})

	return model


def load_optimizer(model, training_steps, args):

	optimizer = AdamW(model.parameters(), lr=args.learning_rate)

	lr_scheduler = get_linear_schedule_with_warmup(
	    optimizer, num_warmup_steps=args.warm_up_steps, num_training_steps=training_steps
	)

	return optimizer, lr_scheduler


def train_epoch(model, train_dataloader, optimizer, lr_scheduler, device, tb_writer, current_epoch):
	"""
		Trains the model for one epoch and returns the average loss
	"""
	# Set train mode
	model.train()

	total_loss = 0.0
	global_step = (current_epoch-1) * len(train_dataloader)
	with tqdm(total=len(train_dataloader), desc="Training epoch") as pbar:
		for batch in train_dataloader:

			optimizer.zero_grad()

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# Forward pass
			outputs = model(input_ids=input_ids,
							attention_mask=attention_mask,
							labels=labels)

			loss = outputs.loss
			total_loss += loss.item()

			# Backpropagation and weight updates
			loss.backward()
			optimizer.step()

			# Logging gradients
			all_gradients_norm = 0
			for name, param in model.named_parameters():
				if param.grad is not None:
					all_gradients_norm += param.grad.norm(2).item()
					if name in LOG_LAYERS:
						tb_writer.add_histogram(f"grads/{name}", param.grad, global_step)
						grad_norm = param.grad.norm(2).item()
						tb_writer.add_scalar(f"grad_norm/{name}", grad_norm, global_step)
			tb_writer.add_scalar("all_gradients_norm", all_gradients_norm, global_step)

			# progress bar
			pbar.update(1)

			global_step += 1

			lr_scheduler.step()

	return total_loss / len(train_dataloader)


def validate_epoch(model, validation_dataloader, device):
	"""
		Validates the model using whole validation data and returns the average loss
	"""
	# Set evaluation mode
	model.eval()

	total_loss = 0.0
	with tqdm(total=len(validation_dataloader), desc="Validating epoch") as pbar:
		for batch in validation_dataloader:

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# Forward pass
			outputs = model(input_ids=input_ids,
							attention_mask=attention_mask,
							labels=labels)

			loss = outputs.loss
			total_loss = loss.item()

			# progress bar
			pbar.update(1)

	return total_loss / len(validation_dataloader)



def train_model(model, dataloaders, args):
	"""
		Trains the model evaluating it over the validation data after each epoch.
		Saves the best model and training statistics.		
	"""

	# Setting up the device
	device = torch.device(args.device)
	model.to(device)

	# Compute total training steps for the optimizer scheduler
	total_train_steps = args.epochs * len(dataloaders['train'])

	# Loading optimizer
	optimizer, lr_scheduler = load_optimizer(model, total_train_steps, args)

	# Starting TensorBoard writer
	writer = SummaryWriter()

	training_statistics = []

	# Training loop
	with tqdm(total=args.epochs, desc="Overall training progress") as pbar:
		for epoch in range(1, args.epochs+1):
			train_loss = train_epoch(model, dataloaders['train'], optimizer, lr_scheduler, device, writer, epoch)
			validation_loss = validate_epoch(model, dataloaders['validation'], device)
			training_statistics.append({
				"epoch": epoch,
				"train loss": train_loss,
				"validation loss": validation_loss
			})

			# Logging epoch metrics using ML FLow
			current_lr = optimizer.param_groups[0]['lr']

			mlflow.log_metric("train_loss", train_loss, step=epoch)
			mlflow.log_metric("validation_loss", validation_loss, step=epoch)
			mlflow.log_metric("learning_rate", current_lr, step=epoch)

			# Logging epoch metrics using tensorboard
			writer.add_scalar("train_loss", train_loss, epoch)
			writer.add_scalar("validation_loss", validation_loss, epoch)
			writer.add_scalar("learning_rate", current_lr, epoch)

			# Logging weights
			for name, param in model.named_parameters():
				if name in LOG_LAYERS:
					writer.add_histogram(f"weights/{name}", param, epoch)
					weight_norm = param.norm(2).item()
					writer.add_scalar(f"weight_norm/{name}", weight_norm, epoch)

			# Saving model in a temporary path, so we can get the best one at the end
			model_output_path = os.path.join(args.output_path, "tmp_models", "epoch_{}".format(epoch))
			os.makedirs(model_output_path, exist_ok=True)
			model.save_pretrained(model_output_path)

			pbar.update(1)

	# Making sure all TensorBoard events are logged and closing the writer
	writer.flush()
	writer.close()



	# Loading best model and deleting the temporary files
	best_epoch = sorted(training_statistics, key=lambda x: x['validation loss'])[0]['epoch']
	best_model_path = os.path.join(args.output_path, "tmp_models", "epoch_{}".format(best_epoch))
	model = DistilBertForSequenceClassification.from_pretrained(best_model_path)

	shutil.rmtree(os.path.join(args.output_path, "tmp_models"))

	return model














