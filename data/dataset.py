import json
import torch
from torch.utils.data import Dataset, DataLoader
from models.tokenizers import RegexTokenizer

GPT_SPECIAL_TOKENS = {'start_sample': '<START_SAMPLE>',
                    'end_sample': '<END_SAMPLE>',
                    'mutation_sep': '<MUT_SEP>',
                    'start_id': '<START_ID>',
                    'end_id': '<END_ID>',
                    'start_pop': '<START_POP>',
                    'end_pop': '<END_POP>',
                    'pad_token': '<PAD>',
                    'unk_token': '<UNK>'}

class GPTDataFormatter:
    """
    """
    def __init__(self, special_tokens=GPT_SPECIAL_TOKENS, custom=False):
        self.special_tokens = special_tokens
        self.custom = custom

    def load_data_from_json(self, json_file_path):
        """
        """
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def parse_mutation(self, mutation_str):
        """
        Extract info from a single mutation in the format 
        "chr:pos:ref>alt_gt"
        """
        chrom, pos, alleles = mutation_str.split(':')
        alleles, gt = alleles.split('_')
        ref, alt = alleles.split('>')
        mut_dict = {'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'gt': gt}
        return mut_dict
    
    def mut_dict_to_str(self, mutation_dict):
        """
        """
        mut_str = f"{mutation_dict['chrom']}:{mutation_dict['pos']}:" \
                    f"{mutation_dict['ref']}>{mutation_dict['alt']}_{mutation_dict['gt']}"
        return mut_str
    
    def mut_str_to_custom(self, generated_mutations):
        """
        """
        # remove tokenizer special tokens
        generated_mutations = generated_mutations.strip(self.special_tokens['start_sample'])
        generated_mutations = generated_mutations.strip(self.special_tokens['end_sample'])
        generated_mutations = generated_mutations.split(self.special_tokens['mutation_sep'])
        custom_formatted_muts = ''
        
        for mut in generated_mutations:
            mut = mut.strip()
            # [CHR-, POS-, REF-, ALT-, GT-]            
            # if not self.custom:
            if len(mut.split(':')) < 3:
            # partial mutations generated eg. at the end of sequence
                continue
            formatted_mut = mut
            
            custom_formatted_muts += f' {formatted_mut}'
            
        return custom_formatted_muts

    def format_sample(self, genotypes, sample_id=None, pop_code=None, pad=None):
        """
        """
        assert genotypes is not None, "Empty genotypes passed."

        parsed_mutations = [self.parse_mutation(mut) for mut in genotypes if mut != '']
        formatted_mutations = [self.mut_dict_to_str(mut) for mut in parsed_mutations if mut != '']

        # start sample sequence with special token
        sample_str = f"{self.special_tokens['start_sample']}"

        if sample_id is not None and not self.custom:
            sample_str += f"{self.special_tokens['start_id']}{sample_id}{self.special_tokens['end_id']}"

        if pop_code is not None and not self.custom:
            sample_str += f"{self.special_tokens['start_pop']}{pop_code}{self.special_tokens['end_pop']}"

        sample_str += f"{self.special_tokens['mutation_sep']}".join(formatted_mutations)
        
        if pad is not None:
            assert isinstance(pad, int), "Pad parameter must be an integer."
            for _ in range(pad):
                sample_str += f"{self.special_tokens['pad_token']}"
        
        # end sample with special token
        sample_str += f"{self.special_tokens['end_sample']}"

        return sample_str
    
    def get_training_corpus(self, data, max_sample_muts=1000,
                            include_pop=False, include_id=False):
        """
        """
        formatted_samples = []
        
        for sample_id, genotypes in data.items():
            sample_mutations = genotypes['genotypes'].split(' ')#[:max_sample_muts]
            # sample_mutations = ' '.join(sample_mutations)

            if not include_id or self.custom:
                sample_id = None
            if include_pop and not self.custom:
                pop_code = genotypes['population']
            else:
                pop_code = None

            formatted_sample = self.format_sample(sample_mutations, sample_id, pop_code)
            formatted_samples.append(formatted_sample)
        
        return formatted_samples
    
    def format_prompt(self, prompt, sample_id, pop_code):
        """
        Assume prompt is in the form "chr:pos:ref>alt_gt chr:pos:..."
        """
        prompt = prompt.split(' ')
        formatted_prompt = self.format_sample(prompt, sample_id, pop_code, pad=None)
        # remove EOS token
        formatted_prompt = formatted_prompt.rstrip(self.special_tokens['end_sample'])
        return formatted_prompt

########################### Data utils for MinGPT ###########################
class MinGPTDataset(Dataset):
    """
    For Causal Language Modelling
    """
    
    def __init__(self, sequences, tokenizer, max_length=4634):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode(sequence, allowed_special='all')
        # print(self.max_length, len(encoding))

        if len(encoding) > self.max_length:
            # print('truncating:\nbefore', len(encoding))
            encoding = encoding[:self.max_length]
            # attention_mask = [1] * len(encoding)
            # print('after', len(encoding))
        else:
            # print('padding:\nbefore', len(encoding))
            pad_token = self.tokenizer.encode('<PAD>', allowed_special='all')[0]
            # pad_token = self.tokenizer.encode('<|endoftext|>', allowed_special='all')
            # attention_mask = [1] * len(encoding) + [0] * (self.max_length - len(encoding))
            encoding.extend([pad_token] * (self.max_length - len(encoding)))
            # print('after', len(encoding))

        inputs = torch.tensor(encoding[:-1], dtype=torch.long)
        labels = torch.tensor(encoding[1:], dtype=torch.long)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {'input_ids': inputs.flatten(),
                # 'attention_mask': attention_mask.flatten(),
                # Same as input_ids for CLM
                'labels': labels.flatten()}

############################# Data utils GPT-2 #############################
class FinetunedGPTDataset(Dataset):
    """
    For Causal Language Modelling
    """
    
    def __init__(self, sequences, tokenizer, max_length=1024):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        encoding = self.tokenizer(sequence,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_length,
                                return_tensors='pt')
        
        return {'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                # Same as input_ids for CLM
                'labels': encoding['input_ids']}