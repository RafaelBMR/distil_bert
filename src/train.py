import argparse
import pathlib
import os

import mlflow

from preprocessing import load_data
from preprocessing import prepare_data

from model import load_model, train_model

from evaluate import test_model


def run(args):

	# Load the dataset
	(data, id2label) = load_data(args)

	# Convert dataset to DistilBERT's input format, and
	# put it in dataloaders
	dataloaders = prepare_data(data, len(id2label), args)

	# Train the model and returns its best version
	model = load_model(args.model_name_or_path, id2label)
	print(model.config)
	for layer_name, layer_params in model.named_parameters():
		print(layer_name, layer_params.size())

	# Set up a mlflow experiment
	mlflow.set_experiment("go-emotions-distilbert")

	with mlflow.start_run():
		# Logging model parameters
		mlflow.log_param("input_max_length", args.max_length)
		mlflow.log_param("initial_learning_rate", args.learning_rate)
		mlflow.log_param("warm_up_steps", args.warm_up_steps)
		mlflow.log_param("batch_size", args.batch_size)
		mlflow.log_param("epochs", args.epochs)

		# Training the model
		model = train_model(model, dataloaders, args)

		# Evaluates best model on the test set
		test_model(model, dataloaders['test'], id2label, args)

	# Saving best model
	model.save_pretrained(os.path.join(args.output_path, "best_model"))






if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset-path", 
						help="Expects a directory with four files: "
							 "\"train.json\" with a list of items, where each item has "
							 "a key \"text\", containing the sample's text, and "
							 "\"labels\" containing a list of ids; \"validation.json\" "
							 "and \"test.json\" in the same format; and \"id2label.json\" "
							 "mapping ids of labels and their names.",
						type=pathlib.Path)

	parser.add_argument("--model-name-or-path", type=str)
	parser.add_argument('--max-length', type=int)

	parser.add_argument("--learning-rate", type=float)
	parser.add_argument("--warm-up-steps", type=int)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument('--epochs', type=int)

	parser.add_argument('--device', type=str, default='cuda')

	parser.add_argument('--output-path', type=pathlib.Path)

	args = parser.parse_args()

	run(args)
