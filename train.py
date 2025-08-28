from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

def main():
    model_name = "distilbert-base-uncased"
    print("Loading dataset...")
    dataset = load_dataset("imdb")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    print("Tokenizing dataset...")
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer...")
    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")
    print("Training complete!")

if __name__ == "__main__":
    main()
