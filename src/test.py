from utils import parse_args
from model import MoEConfig, MoE
from tokenizers import Tokenizer

if __name__=='__main__':
    training_args, args = parse_args('./configs/smoke_test.json')

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    args.vocab_size = tokenizer.get_vocab_size()

    print(args.tokenizer)

    config = MoEConfig(args)
    model = MoE(config=config)