import yaml
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
CONFIG_FILE = "config/config.yaml"

class Initializer:
    def __init__(self):

        model_path = "daveni/twitter-xlm-roberta-emotion-es"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
