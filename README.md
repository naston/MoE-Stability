# MoE-Stability

Steps:
- Implement basic MoE architecture
- Implement simple training loop (The current code is able to work with Lab2 code)
- Find dataset and tokenizer (working) (wikipedia english and huggingface tokenizers BPE?)
- Create training and eval code
- Create Config setup
- Create Fedus Load Balance Loss
- Tensorboard/ WandB monitoring 
- Add in evaluation metrics
- Train and benchmark for testing (this is where I need to be in 2-3 weeks)
- Verify results against papers?
- Create additional loss terms
- Create norm and projected router
- Expert Choice routing?


There is a chance I need to start with say BERT base and then duplicate the FF and add noise for "experts", this may affect results

I have now set up my model to work with the HuggingFace Trainer
    - may have to change output to type ModelOutput
    - This may also mean loss is calculated internally?
I now need to create a HuggingFace training script and then test this
    - use BERT tokenizer and Wikidump preprocessed at like 2022 I think they have
    - I will be testing reporting to tensorboard right after this

Once I am done with this I will be at the step of creating loss terms and monitoring