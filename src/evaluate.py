import os
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import classification_report

import torch

import mlflow

"""
	TODO: salvar report (se ml flow ou tensorboard aceitarem, melhor)
			mas pode ser uma planilha além deles também

"""

def test_model(model, test_dataloader, id2label, args):

	device = torch.device(args.device)

	y_true = []
	y_pred = []

	with tqdm(total=len(test_dataloader), desc="Testing model") as pbar:
		for batch in test_dataloader:

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# Forward pass
			outputs = model(input_ids=input_ids,
							attention_mask=attention_mask,
							labels=labels)

			# Considering 0.5 as threshold for classification
			outputs_probs = torch.sigmoid(outputs.logits)

			y_true.append(labels.detach().cpu())
			y_pred.append(outputs_probs.detach().cpu() > 0.5)

			# progress bar
			pbar.update(1)

	y_true = torch.cat(y_true)
	y_pred = torch.cat(y_pred)

	report = classification_report(y_true=y_true, y_pred=y_pred.float(), digits=4, 
									zero_division=0,
									target_names=[id2label[label_id] for label_id in range(len(id2label))])
	print(report)

	report_dict = classification_report(y_true=y_true, y_pred=y_pred.float(), output_dict=True, 
									zero_division=0,
									target_names=[id2label[label_id] for label_id in range(len(id2label))])

	mlflow.log_metric("Micro F1-Score", report_dict['micro avg']['f1-score'])
	mlflow.log_metric("Macro F1-Score", report_dict['macro avg']['f1-score'])
	mlflow.log_metric("Weighted F1-Score", report_dict['weighted avg']['f1-score'])

	# Saving classification report in a xlsx file
	report_df = pd.DataFrame(report_dict).transpose()
	report_df.to_excel(os.path.join(args.output_path, "Classification_Report_Test_Set.xlsx"), index=False)

	return

