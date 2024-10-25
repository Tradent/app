import logging
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTModel:
    def __init__(self, model_name='bert-base-uncased', config_path=None):
        try:
            if config_path:
                self.config = BertConfig.from_pretrained(config_path)
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertForSequenceClassification.from_pretrained(model_name, config=self.config)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertForSequenceClassification.from_pretrained(model_name)
            logger.info("BERT model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise

    def tokenize(self, texts):
        try:
            return self.tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt')
        except Exception as e:
            logger.error(f"Error tokenizing texts: {e}")
            raise

    def predict(self, inputs):
        try:
            outputs = self.model(**inputs)
            return outputs.logits
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def train(self, dataset_name, output_dir='./results', epochs=3, batch_size=16, learning_rate=2e-5):
        try:
            dataset = load_dataset(dataset_name)
            tokenized_datasets = dataset.map(lambda x: self.tokenize(x['text']), batched=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy='epoch',
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['test'],
            )

            trainer.train()
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save_model(self, save_directory):
        try:
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            logger.info(f"Model and tokenizer saved to {save_directory}.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, load_directory):
        try:
            self.model = BertForSequenceClassification.from_pretrained(load_directory)
            self.tokenizer = BertTokenizer.from_pretrained(load_directory)
            logger.info(f"Model and tokenizer loaded from {load_directory}.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise