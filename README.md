# DistilBERT Classifier

This repository contains code to train and evaluate a DistilBERT for multilabel classification.

## Example of training command

```
python -m src.train \
	--dataset-path "input_data" \
	--model-name-or-path "distilbert-base-uncased" \
	--max-length 128 \
	--learning-rate 5e-5 \
	--warm-up-steps 1000 \
	--batch-size 16 \
	--epochs 4 \
	--device "cuda" \
	--output-path "output"

```