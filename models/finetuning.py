# Finetuning of GPT-2 from pretrained Transformer
import torch
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from data.dataset import FinetunedGPTDataset
from data.data_utils import generate_random_sample, get_running_device

GPT_SPECIAL_TOKENS = {'start_sample': '<START_SAMPLE>',
                    'end_sample': '<END_SAMPLE>',
                    'mutation_sep': '<MUT_SEP>',
                    'start_id': '<START_ID>',
                    'end_id': '<END_ID>',
                    'start_pop': '<START_POP>',
                    'end_pop': '<END_POP>',
                    'pad_token': '<PAD>',
                    'unk_token': '<UNK>'}

def generate_sample(model, tokenizer, formatter, prompt=None, max_sample_length=500, 
                    samples_to_generate=1, skip_special_tokens=False, custom=False,
                    temperature=0.8, return_tensors=False, **kwargs):
    """
    Generate synthetic samples from a (finetuned) GPT-2 model.
    If no prompt is provided, a random sample with 1 mutation is generated.
    Args:
        model: GPT-2 model (finetuned or pretrained)
        tokenizer: GPT-2 tokenizer
        formatter: Formatter object to format input/output strings
        prompt (optional): input prompt string to condition generation
        max_sample_length: maximum number of tokens to generate for each sample
                            (Defaults to 500)
        samples_to_generate: number of samples to generate (Defaults to 1)
        skip_special_tokens: whether to skip special tokens in the output
                            (Defaults to False)
        custom: whether to convert output to custom format (Defaults to False)
        temperature: sampling temperature (Defaults to 0.8)
        return_tensors: whether to return raw token tensors (Defaults to False)
        **kwargs: additional arguments

    Returns:
        list: generated samples as strings or tensor token ids
    """
    GPT_SPECIAL_TOKENS = {'start_sample': '<START_SAMPLE>',
                    'end_sample': '<END_SAMPLE>',
                    'mutation_sep': '<MUT_SEP>',
                    'start_id': '<START_ID>',
                    'end_id': '<END_ID>',
                    'start_pop': '<START_POP>',
                    'end_pop': '<END_POP>',
                    'pad_token': '<PAD>',
                    'unk_token': '<UNK>'}
    special_tokens = kwargs.get('special_tokens', GPT_SPECIAL_TOKENS)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    save_path = kwargs.get('save_path', None)
    
    print('Generating on', device)
    if prompt is None:
        # ['<END_ID>', '<MUT_SEP>','<PAD>','<START_ID>','<START_POP>','<END_POP>','<START_SAMPLE>','<UNK>','<END_SAMPLE>']
        # prompt = f'{self.tokenizer.additional_special_tokens[6]}'  # <START_SAMPLE>
        # prompt = f'{special_tokens['start_sample']}'  # <START_SAMPLE>
        prompt = generate_random_sample(num_mutations=1)
        pop_code = None
    else:
        pop_code = None
        for pop in ['EUR', 'EAS', 'AFR', 'SAS', 'AMR']:
            if pop in prompt:
                pop_code = pop
                break

    prompt = formatter.format_prompt(prompt, sample_id=None, pop_code=pop_code)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # model.to(device)
    model.eval()  
    with torch.no_grad():
        pad_tok = tokenizer.pad_token_id
        eos_tok = tokenizer.encode(special_tokens['end_sample'])[0]

        outputs = model.generate(input_ids.to(device),
                                max_new_tokens=max_sample_length,
                                num_return_sequences=samples_to_generate,
                                temperature=temperature,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=pad_tok,
                                eos_token_id=eos_tok)

    if return_tensors:
        return outputs
    else:
        # Decode generated sequences
        generated_samples = []
        for output in outputs:
            sample = tokenizer.decode(output, 
                                        skip_special_tokens=skip_special_tokens)
            if not custom:
                sample = formatter.mut_str_to_custom(sample)
            generated_samples.append(sample)

        if save_path is not None:
            generated_samples = {f'synth_{i+1}':sample for i, sample in enumerate(generated_samples)}
            with open(save_path, 'w+') as f:
                json.dump(generated_samples, f, sort_keys=False, indent=4)

        return generated_samples

############################# Finetuning HF GPT-2 #############################
class FinetuningTrainer:
    """
    Class to handle finetuning of GPT-2 from pretrained Transformer
    model using Hugging Face Trainer API.
    """
    def __init__(self, output_dir="models/saved/GPT",
                model_name='gpt2',
                special_tokens=GPT_SPECIAL_TOKENS,
                use_privacy=False,
                **kwargs):
        self.model_name = model_name
        self.output_dir = output_dir
        
        self.use_privacy = use_privacy
        self.target_epsilon = 3.0
        self.target_delta = 1e-5
        self.max_grad_norm = 1.0
        self.noise_multiplier = 1.0

        self.cuda_available = torch.cuda.is_available()
        self.device = kwargs.get('device', 'cuda' if self.cuda_available else 'cpu')
        # enforce no gpu if device manually set to 'cpu'
        if self.device == 'cpu':
            self.cuda_available = False
        # self.device = get_running_device()
        print('Running on', self.device)

        self.special_tokens = special_tokens
        self.setup_tokenizer_and_model()

    def setup_tokenizer_and_model(self):
        """
        Load GPT-2 tokenizer and model from pretrained Transformer.
        Add special tokens to tokenizer and resize model embeddings.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name, gradient_checkpointing=True)
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens':
                               list(self.special_tokens.values())}
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.special_tokens['pad_token']
            special_tokens_dict['pad_token'] = self.special_tokens['pad_token']
        
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Load model from pretrained based on model name (gpt2, gpt2-medium, gpt2-large)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Resize token embeddings to account for new tokens
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def setup_training_data(self, dataset, train_val_split=0.3,
                            max_length=1000):
        """
        Prepare training and evaluation datasets.

        Args:
            dataset: list of input sequences (strings)
            train_val_split: ratio of data to use for validation (Defaults to 0.3)
            max_length: maximum sequence length for tokenization (Defaults to 1000)

        Returns:
            train_dataset: training dataset FinetunedGPTDataset object
            eval_dataset: evaluation dataset FinetunedGPTDataset object
        """
        # Create split datasets
        train_sequences, val_sequences = train_test_split(
                                        dataset, 
                                        test_size=train_val_split, 
                                        random_state=42)
        train_dataset = FinetunedGPTDataset(train_sequences, 
                                        self.tokenizer, 
                                        max_length)
        eval_dataset = FinetunedGPTDataset(val_sequences,
                                        self.tokenizer,
                                        max_length)
        return train_dataset, eval_dataset
    
    def setup_trainer(self, epochs=5, train_batch_size=4, eval_batch_size=4,
                            learning_rate=5e-5, weight_decay=0.01):
        """
        Setup training arguments for Hugging Face or dp-transformers Trainer.
        Handles both standard and differentially private training.

        Args:
            epochs: number of training epochs (Defaults to 5)
            train_batch_size: training batch size (Defaults to 4)
            eval_batch_size: evaluation batch size (Defaults to 4)
            learning_rate: learning rate (Defaults to 5e-5)
            weight_decay: weight decay (Defaults to 0.01)

        Returns:
            training_args: TrainingArguments or dp_transformers.TrainingArguments
                         object
        """ 
        # Set training args
        if self.use_privacy:
            import dp_transformers
            training_args = dp_transformers.TrainingArguments(output_dir=self.output_dir,
                                        overwrite_output_dir=True,
                                        num_train_epochs=epochs,
                                        warmup_steps=100,
                                        per_device_train_batch_size=train_batch_size,
                                        per_device_eval_batch_size=eval_batch_size,
                                        learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        # report_to='wandb',
                                        logging_dir=f'{self.output_dir}/logs',
                                        logging_steps=100,
                                        eval_steps=50,
                                        save_steps=100,
                                        save_total_limit=2,
                                        prediction_loss_only=True,
                                        remove_unused_columns=False,
                                        dataloader_pin_memory=False,
                                        gradient_accumulation_steps=4,) # for memory
        else:
            training_args = TrainingArguments(output_dir=self.output_dir,
                                        overwrite_output_dir=True,
                                        num_train_epochs=epochs,
                                        per_device_train_batch_size=train_batch_size,
                                        per_device_eval_batch_size=eval_batch_size,
                                        warmup_steps=100,
                                        learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        # report_to=None,
                                        logging_dir=f'{self.output_dir}/logs',
                                        logging_steps=100,
                                        eval_steps=50,
                                        save_steps=100,
                                        save_total_limit=2,
                                        prediction_loss_only=True,
                                        remove_unused_columns=False,
                                        dataloader_pin_memory=False,
                                        gradient_accumulation_steps=2, # for memory
                                        fp16=self.cuda_available)
        return training_args
    
    def setup_differential_privacy(self):
        """
        Setup differential privacy arguments and data collator for dp-transformers Trainer.
        """
        import dp_transformers    
        print("Setting up differential privacy...", end='') 

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self.tokenizer)

        privacy_args = dp_transformers.PrivacyArguments(
            target_epsilon = self.target_epsilon,
            target_delta = self.target_delta,
            # noise_multiplier = self.noise_multiplier,
            per_sample_max_grad_norm = self.max_grad_norm,
        )

        print(f"Privacy Budget: ε={self.target_epsilon}, δ={self.target_delta}")
        return privacy_args, data_collator
    
    def train_model(self, train_dataset, eval_dataset, training_args):
        """
        Train the model with the given datasets and training arguments.

        Args:
            train_dataset: training dataset FinetunedGPTDataset object
            eval_dataset: evaluation dataset FinetunedGPTDataset object
            training_args: TrainingArguments or dp_transformers.TrainingArguments

        Returns:
            trainer: trained Trainer or dp_transformers.OpacusDPTrainer object
        """
        if self.use_privacy:
            import dp_transformers
            # from dp_transformers.grad_sample.transformers import conv_1d

            privacy_args, dp_data_collator = self.setup_differential_privacy()

            trainer = dp_transformers.dp_utils.OpacusDPTrainer(
                     args=training_args,
                     model=self.model,
                     train_dataset=train_dataset,
                     eval_dataset=eval_dataset,
                     data_collator=dp_data_collator,
                     privacy_args=privacy_args,
                     tokenizer=self.tokenizer)
            # trainer.data_collator = dp_data_collator
        else:
            data_collator = DataCollatorForLanguageModeling(
                                tokenizer=self.tokenizer,
                                mlm=False) # False for CLM
            # Initialize trainer
            trainer = Trainer(model=self.model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer=self.tokenizer)
        
        self.trainer = trainer
        
        # Train the model
        print("Starting training...")
        torch.cuda.empty_cache()
        # trainer.train()

        try:
            trainer.train()
        finally:
            if self.use_privacy:
                eps_prv = trainer.get_prv_epsilon()
                eps_rdp = trainer.get_rdp_epsilon()
                print(f"Final privacy cost: rdp (prv) = {eps_rdp} ({eps_prv})")

        # if self.use_privacy:
        #     privacy_engine = trainer.privacy_engine
        #     epsilon = privacy_engine.get_epsilon(delta=1e-5)
        #     print(f"Final privacy cost: ε = {epsilon:.2f}")

        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def evaluate_model(self, trainer):
        """
        Evaluate the trained model using the trainer's evaluate method.

        Args:
            trainer: trained Trainer or dp_transformers.OpacusDPTrainer object
        """
        print("Evaluating model...", end='')
        eval_results = trainer.evaluate()
        
        print(f"Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
        
        return eval_results

##################### Pretrained HF GPT-2 (no finetuning) #####################
from transformers import AutoTokenizer, pipeline
import json

class PretrainedGPT:
    """
    Class to handle text generation from pretrained benchmarking GPT-2 model
    using Hugging Face pipeline API.
    """
    def __init__(self, model_name='gpt2', **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       device=self.device)
        self.model = pipeline('text-generation', model=model_name,
                            device=self.device)

    def generate(self, prompt, samples_to_generate=1, max_length=500, save_path=None):
        generated_samples = self.model(prompt, 
                                    max_length=max_length,
                                    num_return_sequences=samples_to_generate)

        generated_samples = {f'synth_{i+1}': generated_samples[i]['generated_text'] 
                            for i in range(len(generated_samples))}

        if save_path is not None:
            with open(save_path, 'w+') as f:
                json.dump(generated_samples, f, sort_keys=False, indent=4)
        
        return generated_samples