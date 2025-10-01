# Defining which layers to log

LOG_LAYERS = [
	"distilbert.transformer.layer.0.attention.out_lin.weight",
	"distilbert.transformer.layer.1.attention.out_lin.weight",
	"distilbert.transformer.layer.2.attention.out_lin.weight",
	"distilbert.transformer.layer.3.attention.out_lin.weight",
	"distilbert.transformer.layer.4.attention.out_lin.weight",
	"distilbert.transformer.layer.5.attention.out_lin.weight",
	"pre_classifier.weight",
	"pre_classifier.bias",
	"classifier.weight",
	"classifier.bias"
]