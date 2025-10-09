import os
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import classification_report

import base64
from io import BytesIO

import seaborn as sns
import matplotlib.pyplot as plt

import torch

import mlflow

from jinja2 import Environment, FileSystemLoader


def save_report(report_df, args):
    """
        Saves an HTML file containing overall metrics and per class metrics
    """

    n_samples = report_df.loc['weighted avg']['support']

    overall_metrics = []
    for average_type in ['weighted avg', 'macro avg', 'micro avg', 'samples avg']:
        overall_metrics.append({
            "average_type": average_type,
            "f1": report_df.loc[average_type]['f1-score'] * 100,
            "precision": report_df.loc[average_type]['precision'] * 100,
            "recall": report_df.loc[average_type]['recall'] * 100,
        })

    classes_order = list(report_df.sort_values("support", ascending=False).index)
    df_plot = []
    for class_name, metrics in report_df.iterrows():
        if class_name in ['weighted avg', 'macro avg', 'micro avg', 'samples avg']:
            classes_order.remove(class_name)
            continue
        for metric in ['f1-score', 'precision', 'recall']:
            df_plot.append({
                "Class name": class_name,
                "Metric": metric,
                "Value": metrics[metric]
            })
    df_plot = pd.DataFrame(df_plot)

    # F1-score plot
    fig = plt.figure(figsize=(10, len(classes_order)/2))

    sns.barplot(
        x='Value',
        y='Class name',
        hue='Metric',
        data=df_plot,
        hue_order=['f1-score'],
        order=classes_order
    )

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded_f1 = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close(fig)

    # Precision and recall plot
    fig = plt.figure(figsize=(10, len(classes_order)/2))

    sns.barplot(
        x='Value',
        y='Class name',
        hue='Metric',
        data=df_plot,
        hue_order=['precision', 'recall'],
        order=classes_order
    )

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded_pr = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close(fig)

    # Rendering template and saving html file
    data = {
        "n_samples": n_samples,
        "metrics": overall_metrics,
        "f1_plot": encoded_f1,
        "pr_plot": encoded_pr
    }

    loader = FileSystemLoader("templates/")
    env = Environment(loader=loader)
    template = env.get_template("report.html")
    output = template.render(data)

    with open(os.path.join(args.output_path, "report.html"), mode='w') as f:
        f.write(output)

    return


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

    save_report(report_df, args)

    return

