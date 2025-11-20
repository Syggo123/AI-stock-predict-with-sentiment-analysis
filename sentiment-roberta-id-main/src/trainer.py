from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import configparser


class SentimentTrainer:
    def __init__(self, config_file):
        """
        Initializes the class with the given `config_file`.

        :param config_file: The path to the configuration file.
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        notebook_login()

        self.dataset = load_dataset("csv", data_files={
            "train": "sentiment-roberta-id/src/sent_train.csv",
            "validation": "sentiment-roberta-id/src/sent_valid.csv"
        })

        self.model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )

        self.preprocess_dataset()

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.training_args = TrainingArguments(
            output_dir=self.config.get("TrainingConfig", "output_dir"),
            overwrite_output_dir=self.config.getboolean("TrainingConfig", "overwrite_output_dir"),
            evaluation_strategy=self.config.get("TrainingConfig", "evaluation_strategy"),
            save_strategy=self.config.get("TrainingConfig", "save_strategy"),
            learning_rate=self.config.getfloat("TrainingConfig", "learning_rate"),
            per_device_train_batch_size=self.config.getint("TrainingConfig", "per_device_train_batch_size"),
            per_device_eval_batch_size=self.config.getint("TrainingConfig", "per_device_eval_batch_size"),
            save_steps=self.config.getint("TrainingConfig", "save_steps"),
            save_total_limit=self.config.getint("TrainingConfig", "save_total_limit"),
            num_train_epochs=self.config.getint("TrainingConfig", "num_train_epochs"),
            metric_for_best_model=self.config.get("TrainingConfig", "metric_for_best_model"),
            greater_is_better=self.config.getboolean("TrainingConfig", "greater_is_better"),
            load_best_model_at_end=self.config.getboolean("TrainingConfig", "load_best_model_at_end"),
            logging_steps=self.config.getint("TrainingConfig", "logging_steps"),
            weight_decay=self.config.getfloat("TrainingConfig", "weight_decay"),
            report_to=self.config.get("TrainingConfig", "report_to"),
            push_to_hub=self.config.getboolean("TrainingConfig", "push_to_hub"),
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

    def preprocess_function(self, examples):
        """
        Preprocesses the given examples using the tokenizer.

        Args:
            examples (dict): The examples to be preprocessed.

        Returns:
            dict: The preprocessed examples.
        """
        return self.tokenizer(examples["text"], truncation=True)

    def preprocess_dataset(self):
        """
        Preprocesses the dataset.

        This function tokenizes the dataset using the `preprocess_function` and stores the tokenized dataset in the `tokenized_dataset` attribute.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        """
        self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True)

    def train(self):
        """
        Train the model using the trainer object.

        Parameters:
            None

        Returns:
            None
        """
        self.trainer.train()

    def push_to_hub(self):
        """
        Pushes the current state of the trainer to the hub.

        This function calls the `push_to_hub` method of the `trainer` object.

        Parameters:
            None

        Returns:
            None
        """
        self.trainer.push_to_hub()


if __name__ == "__main__":
    trainer = SentimentTrainer("sentiment-roberta-id/src/config.conf")
    trainer.train()
    trainer.push_to_hub()