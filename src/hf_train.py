import datasets
from model import MoE, MoEConfig
from utils import parse_args, training_metrics
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from datasets import load_dataset
from trainer import MoETrainer

def train_lm():
    training_args, args = parse_args(file_path='./configs/smoke_test.json')
    tokenizer = Tokenizer.from_pretrained(args.tokenizer)
    args.vocab_size = tokenizer.get_vocab_size()

    config = MoEConfig(args)
    model = MoE(config)

    dataset = load_dataset(args.dataset, args.dataset_version)

    """
    training_args = TrainingArguments(
        output_dir="./models",
        overwrite_output_dir=True,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    ) 
    """

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        #report_to=['tensorboard'],
        #compute_metrics=training_metrics,
    )

    trainer.train()
    trainer.save_model('./models')

if __name__=='__main__':
    train_lm()