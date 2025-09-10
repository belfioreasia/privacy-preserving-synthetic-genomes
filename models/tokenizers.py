import os
import json
import tqdm
import re
import torch
import pandas as pd
import numpy as np
from data.data_utils import SEQUENCE_PATTERN, UNK_PATTERN, load_corpus

TRAIN_CORPUS = 'data/sources/test/test_corpus_chr22.json'
SAVE_TOKENIZER_PATH = 'models/saved/tokenizers'
SPECIAL_TOKENS = ["[UNK]", "[SEP]", "[MASK]"]
# SPECIAL_TOKENS = ['<START_SAMPLE>', '<END_SAMPLE>', '<MUT_SEP>',
#                 '<START_ID>', '<END_ID>', '<START_POP>',
#                 '<END_POP>', '<PAD>', '<UNK>']

VOCAB_INIT = {
        # dividers and delimiters
        ':': 0,  # between chromosome, position and alleles
        '_': 1,  # between mutation and sample alleles
        '>': 2,  # between reference and alternative alleles
        '|': 3,  # for sample genotype (phased alleles)
        '/': 4,  # for sample genotype (non-phased alleles)
        # nucleotides
        'A': 5,
        'C': 6,
        'G': 7,
        'T': 8,
        # digits for sample genotypes and (chr/pos) numbering
        '0': 9,
        '1': 10, 
        '2': 11,
        '3': 12,
        '4': 13,
        '5': 14,
        '6': 15,
        '7': 16,
        '8': 17,
        '9': 18, 
        '[UNK]': 19,
}

def iterate_corpus(corpus):
    """
    Iterate over the corpus and yield each sequence.

    Args:
        corpups (dict): A dictionary representing the corpus.

    Yields:
        str: Each sequence in the corpus.
    """
    for id,seq in tqdm.tqdm(corpus.items(), desc="Iterating corpus"):
        if id in seq: # remove the sample id from the sequence (if present)
            seq = seq.strip(f'{id} ') 
        yield seq

def preprocess_corpus_text(corpus, pattern = SEQUENCE_PATTERN, 
                   separators = {'[SEP]': ' '}, save_path = None):
    """
    Convert the corpus to a single text string.
    Note that Ġ is a particularity of the GPT-2 BPE implementation
    and it is used to separate words in the text (i.e. replace spaces).

    Args:
        corpus (dict): A dictionary representing the corpus.
        sep (dict): Separataors to use within the sentence. Must be a dict
                    with keys 'sep_name' and 'sep_token'. 
                    E.g., { '[BOS]': '<|startofsample|>',
                            '[EOS]': '<|endofsample|>', 
                            '[SEP]': 'Ġ' }

    Returns:
        str: A single text string representing the corpus.
    """
    if '[SEP]' not in separators.keys(): 
        print("Separators MUST contain '[SEP]' key for separation of mutations.")
        separators = {'[SEP]': ' '}
        
    processed_text = []
    substring = []

    for sample_gts in tqdm.tqdm(corpus.values(), desc="Processing corpus"):
        if '[BOS]' in separators.keys():
            processed_text.append(separators['[BOS]'])
        for i, char in enumerate(sample_gts):
            if char == " " and i != 0:
                # do not include sample id AND only include valid substrings
                if re.fullmatch(pattern, "".join(substring)):
                    # print("Adding substring:", "".join(substring))
                    processed_text.extend([tok for tok in substring])
                    processed_text.append(separators['[SEP]'])
                substring = []
                # print("So far:", "".join(processed_text))
            if char != " ":
                substring.append(char)
        if '[EOS]' in separators.keys():
            processed_text.append(separators['[EOS]'])
    processed_text = "".join(processed_text)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(processed_text)
    return processed_text

def tokenize_dataset(sample_genotypes, tokenizer,
                    population_file="data/sources/1000GP/sample_lookup.csv",
                    labels=False, 
                    label_names=None, 
                    has_sample_names=False, 
                    tensorize=False):
    tokenized_gts = []
    if labels:
        populations = pd.read_csv(population_file, sep="\t")
        sample_pops = []
    sample_ids = list(sample_genotypes.keys())
    for sample_id in tqdm.tqdm(sample_ids, desc="Tokenizing genotypes"):
        gt_tok_ids = tokenizer.encode(sample_genotypes[sample_id])
        if tensorize:
            tokenized_gts.append(torch.tensor(gt_tok_ids, dtype=torch.long))
        else:
            tokenized_gts.append(np.array(gt_tok_ids, dtype=np.long))
        if labels:
            sample_pops.append(populations[populations['Sample'] == sample_id]['Population'].values[0])
    serialised_dataset = pd.DataFrame(sample_ids, columns=['sample_id'])
    serialised_dataset['tokenized_genotypes'] = tokenized_gts
    if has_sample_names:
        serialised_dataset['genotypes'] = [gt.strip(sample_ids[i]) for i,gt in enumerate(list(sample_genotypes.values()))]
    else:
        serialised_dataset['genotypes'] = list(sample_genotypes.values())
    serialised_dataset.set_index('sample_id', inplace=True)
    if labels:
        serialised_dataset[label_names] = sample_pops
    return serialised_dataset

############################ Regex BPE Tokenization ############################
# from https://github.com/karpathy/minbpe/tree/master
import unicodedata

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class RegexTokenizer():
    """
    Minimal (byte-level) Byte Pair Encoding tokenizer from
    https://github.com/karpathy/minbpe/tree/master

    Algorithmically follows along the GPT tokenizer:
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    - uses a regex-based splitting pattern.
    - handles optional special tokens.
    """
    def __init__(self, pattern=SEQUENCE_PATTERN, special_tokens=None):
        """
        - pattern: string pattern for tokenization
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 4097,
                    'Ġ': 4098}
        """
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        if special_tokens is not None:
            self.register_special_tokens(special_tokens)
        
        self.vocab = self._build_vocab() # int -> bytes
        self.pattern = SEQUENCE_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        print(f"Found {len(text_chunks)} formatted mutations", end=', ')

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        print(f"encoded into {sum(len(chunk) for chunk in ids)} byte tokens")

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        training_log = tqdm.tqdm(range(num_merges), desc="Training Regex BPE tokenizer",)
        for i in training_log:
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                training_log.set_postfix_str(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        text_chunks = re.findall(self.compiled_pattern, text)
        return ' '.join(text_chunks)

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        This function handles encoding of special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix="models/saved/tokenizers/Regex/RegexTokenizer"):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file="models/saved/tokenizers/Regex/RegexTokenizer.model"):
        """
        Load the saved tokenizer from a .model file.
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

############################### BPE Tokenization ###############################
from collections import Counter, deque
from functools import lru_cache
class BPEVCFTokenizer():
    """
    Adapted From https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
    and 
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb
    """
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.tok_to_id = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

        # For the official OpenAI GPT-2 merges, use a rank dict:
        #  of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}

        self.pattern = SEQUENCE_PATTERN
        self.unk_pattern = UNK_PATTERN  # for decoding to handle unk tokens
        self._allele_pattern = r"(?:\[UNK\]|(?:[ATCG]+(?:/[ATCG]+)*))"

    def __str__(self):
        tokenizer_str = "BPE-based VCF Tokenizer():\n"
        tokenizer_str += f"   Vocabulary Size: {self.vocab_size}\n"
        tokenizer_str += f"   Recognized Pattern: {self.pattern}\n"
        return tokenizer_str
    
    def train(self, corpus, vocab_size, allowed_special={"<|endoftext|>", 'Ġ'}):
        
        def special_token_lookup(special_tokens):
            """
            Map each special token to its corresponding BPE token.
            This is a workaround for the BPE tokenizer to handle special tokens
            that are not part of the BPE vocabulary.
            """
            lookup = {'[EOS]': '<|endoftext|>',
                    '[BOS]': '<|startoftext|>', 
                    '[SEP]': 'Ġ' }
            matches = set(lookup.values()).intersection(special_tokens)
            allowed_special = {}
            for token in matches:
                if token in lookup.values():
                    key = list(lookup.keys())[list(lookup.values()).index(token)]
                    allowed_special[key] = token
            if allowed_special is not None:
                return allowed_special
            else:
                return {'[SEP]': 'Ġ'}

        """
        Train the BPE tokenizer from scratch.
        """
        print("Loading training corpus...")
        if corpus.endswith(".json"):
            corpus = load_corpus(corpus)
            separators = special_token_lookup(allowed_special)
            print(separators)
            processed_text = preprocess_corpus_text(corpus, pattern=self.pattern, 
                                                    separators=separators,
                                                    save_path="data/sources/bpe_processed_corpus.txt")
        else:
            # If corpus is a string/text file, assume it's already preprocessed
            with open(corpus, 'r', encoding='utf-8') as file:
                corpus = file.read()
            processed_text = corpus
        
        print("Building vocabulary from processed text...")

        # Initialize vocab with unique characters
        unique_chars = [tok for tok in VOCAB_INIT]
        unique_chars.extend(char for char in sorted(set(processed_text))
                            if char not in unique_chars)
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.tok_to_id = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.tok_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.tok_to_id[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.tok_to_id[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs       
        for new_id in  tqdm.tqdm(range(len(self.vocab), vocab_size), 
                                 desc="Finding frequent pairs", position=1, leave=False):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in tqdm.tqdm(self.bpe_merges.items(), 
                                            desc="Building the vocabulary with merged tokens", 
                                            position=2, leave=False): 
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.tok_to_id[merged_token] = new_id

    def load_vocab_and_merges_from_file(self, vocab_path='models/saved/tokenizers/BPE/vocab.json', 
                                          bpe_merges_path='models/saved/tokenizers/BPE/vocab.bpe'):
        """
        Load pre-trained vocabulary and BPE merges.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Convert loaded vocabulary to correct format
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.tok_to_id = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character without adding a new token
        if "\n" not in self.tok_to_id:
            # Use an existing token ID as a placeholder for '\n'
            # Preferentially use "<|endoftext|>" if available
            fallback_token = next((token for token in ["<|endoftext|>", "Ġ", ""] if token in self.tok_to_id), None)
            if fallback_token is not None:
                newline_token_id = self.tok_to_id[fallback_token]
            else:
                # If no fallback token is available, raise an error
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")

            self.tok_to_id["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"

        # Load GPT-2 merges and store them with an assigned "rank"
        self.bpe_ranks = {}  # reset ranks
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            rank = 0
            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    # If token1 or token2 not in vocab, skip
                    if token1 in self.tok_to_id and token2 in self.tok_to_id:
                        self.bpe_ranks[(token1, token2)] = rank
                        rank += 1
                    else:
                        print(f"Skipping pair {pair} as one token is not in the vocabulary.")

    def encode(self, text, allowed_special=None):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.
    
        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.
    
        Returns:
            List of token IDs.
        """
        import re
    
        token_ids = []
    
        # If special token handling is enabled
        if allowed_special is not None and len(allowed_special) > 0:
            # Build regex to match allowed special tokens
            special_pattern = (
                "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            )
    
            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None))  # Encode prefix without special handling
    
                special_token = match.group(0)
                if special_token in self.tok_to_id:
                    token_ids.append(self.tok_to_id[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()
    
            text = text[last_index:]  # Remaining part to process normally
    
            # Check if any disallowed special tokens are in the remainder
            disallowed = [
                tok for tok in self.tok_to_id
                if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
            ]
            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
    
        # If no special tokens, or remaining text after special token split:
        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)
    
        for token in tokens:
            if token in self.tok_to_id:
                token_ids.append(self.tok_to_id[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))
    
        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.tok_to_id.get(char, '[UNK]') for char in token]
        if '[UNK]' in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid == '[UNK]']
            print(f"WARNING: Characters not found in vocab: {missing_chars}")


        # If we haven't loaded OpenAI's GPT-2 merges, use my approach
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        # Uncomment for educational purposes:
                        # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                        i += 2  # Skip the next token as it's merged
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            # Find the pair with the best (lowest) rank
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break

            # Merge all occurrences of that pair
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)  # merged symbol
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = [self.tok_to_id[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "  # Add space if not present before a newline
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.tok_to_id = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.tok_to_id.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced

############################# Manual Tokenization #############################
class ManualTokenizer:
    """
    """
    def __init__(self, subtokenize=True, 
                 special_tokens=SPECIAL_TOKENS, 
                 vocab_init=VOCAB_INIT):
        self.pattern = SEQUENCE_PATTERN
        self.unk_pattern = UNK_PATTERN

        self.vocab = vocab_init
        self.vocab_size = len(self.vocab)
        # self._extend_vocab(special_tokens)
        self.subtokenize = subtokenize

        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __str__(self):
        tokenizer_str = "ManualTokenizer():\n"
        tokenizer_str += f"   Subtokenize: {self.subtokenize}\n"
        tokenizer_str += f"   Vocabulary Size: {self.vocab_size}\n"
        tokenizer_str += f"   Recognized Pattern: {self.pattern}\n"
        return tokenizer_str

    def _extend_vocab(self, tokens):
        """
        Extend the vocabulary with new tokens.
        
        Args:
            tokens (list): A list of tokens to add to the vocabulary.
        """
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1

    def _subtokenize_alleles(self, mutation):
        """
        Manually subtokenize a token in the form 'ref>alt' or 'a1|a2'.

        Args:
            mutation (str): A string representing a mutation.

        Returns:
            list: A list of tokens representing the mutation.
        """
        if '>' in mutation:
            # split by '>' for reference and alternative alleles
            split_by = '>'
        elif '|' in mutation:
            # split by '|' for phased sample alleles
            split_by = '|'
        elif '/' in mutation:
            # split by '/' for not phased genotype
            split_by = '/'
        else:
            raise ValueError(f"Invalid mutation format: {mutation}")
        allele_1, allele_2 = mutation.split(split_by)
        return allele_1, split_by, allele_2
    
    def _subtokenize_digits(self, mutation):
        """
        Manually subtokenize a long string of digits into individual tokens.
        EG: '24982470' -> ['2', '4', '9', '8', '2', '4', '7', '0'].

        Args:
            mutation (str): A string representing a mutation.

        Returns:
            list: A list of tokens representing the long number.
        """
        for digit in mutation:
            if digit not in self.vocab:
                self._extend_vocab([digit])
        return [digit for digit in mutation]

    def _tokenize(self, sequence, tokens = [], bad_muts = [], verbose = False):
        """
        Tokenize a sequence of mutations in the form 'chr:pos:ref>alt_a1|a2 
        chr:pos:ref>alt_a1|a2...'.
        """
        tokens = []
        for mut in sequence.split(' '): # get individual mutations
            if not re.fullmatch(self.pattern, mut):
                # if the mutation does not match the expected pattern, skip it
                # print(f"Skipping invalid mutation format: {mut}")
                continue
            else: 
                mut_split = mut.split(':')
                if '<' in mut_split[2] or len(mut_split) != 3: # invalid mutation
                    # print(f"Skipping invalid mutation format: {mut}")
                    mut = mut.split('_')[0]
                    if mut not in bad_muts:
                        bad_muts.append(mut)
                else:
                    chrom, pos, alleles = mut.split(':') # separate into chr, pos, ref>alt_a1|a2
                    tokens.extend(self._subtokenize_digits(chrom))
                    tokens.extend(':') 
                    tokens.extend(self._subtokenize_digits(pos))
                    # tokens.extend([chrom, ':', pos])
                    # extend vocabulary for new tokens
                    self._extend_vocab([chrom, pos])
                    
                    mut_alleles, sample_gt = alleles.split('_') # split into ref>alt and a1|a2
                    # treat each allele in mutation specification and sample
                    # genotype as separate tokens
                    # EG: '1:545363:T>C_0|0' -> ['1', '545363', 'T', '>', 'C', '0', '0']
                    ref, _, alt = self._subtokenize_alleles(mut_alleles)
                    allele_1, split_by, allele_2 = self._subtokenize_alleles(sample_gt)
                    tokens.extend([ref, '>', alt, allele_1, split_by, allele_2])
                    self._extend_vocab([ref, alt, allele_1, split_by, allele_2])
        if verbose:
            print(f"Tokenized {len(tokens)} tokens from the sequence.")
            if len(bad_muts) > 0:
                print(f"Bad Mutations: {bad_muts}")
        return tokens, bad_muts
    
    def encode(self, sequence):
        """
        Encode a sequence of mutations into token IDs.

        Args:
            sequence (str): A string representing a sequence of mutations.

        Returns:
            list: A list of token IDs representing the sequence.
        """
        tokens, _ = self._tokenize(sequence)
        token_ids = [self.token_to_id.get(token, '[UNK]') for token in tokens]
        return token_ids
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs into a sequence of mutations.

        Args:
            token_ids (list): A list of token IDs representing a sequence.

        Returns:
            str: A string representing the decoded sequence.
        """
        # id_to_token = {v: k for k, v in self.vocab.items()}
        # sequence = ''
        # tok_num = 0 # token number within the sequence (mutation)
        # for tok_id in token_ids:
        #     sequence += id_to_token.get(tok_id, '[UNK]')
        #     if tok_num == 2: # after chr:pos
        #         sequence += ':'
        #     if tok_num == 5: # after chr:pos:ref>alt
        #         sequence += '_'
        #     if tok_num == 8:
        #         sequence += ' ' # separate each mutation
        #         tok_num = -1 # reset in-sequence token counter
        #     tok_num += 1
        sequence = ''
        subsequence = ''
        gt_done = False
        for tok_id in token_ids:
            recon_token = self.id_to_token.get(tok_id, '[UNK]')
            if re.match(r"(?:\[UNK\]|(?:[ATCG]+(?:/[ATCG]+)*))", recon_token):
                if not gt_done:
                    subsequence += ':' + recon_token
                else:
                    subsequence += recon_token + '_'
                gt_done = not gt_done
            elif recon_token == '[UNK]':
                subsequence += ':[UNK]_'
            else:
                subsequence += recon_token
            if re.fullmatch(self.unk_pattern, subsequence):
                sequence += subsequence + ' '
                subsequence = ''
            
        return sequence
    
    def tokenize_corpus(self, corpus):
        """
        Manually tokenize a sequence of sentencess with mutations in the 
        form 'chr:pos:ref>alt_a1|a2'.

        Args:
            corpus (dict): A dictionary representing a corpus of mutations.

        Returns:
            list: A list of tokens representing the sequence.
        """
        tokens = []
        # self.vocab = []
        # self.vocab.extend([tok for tok in SPECIAL_TOKENS])
        # self.vocab.extend([':'])  # add token for chr and pos separation
        # if subtokenize:
        #     self.vocab.extend(['>'])  # add token for ref to alt separation
        #     self.vocab.extend(['|'])  # add token for phased genotype separation
        #     self.vocab.extend(['/'])  # add token for not phased genotype separation

        # tokenize all sequences in the corpus
        tokenization_log = tqdm.tqdm(corpus.items(), desc="Tokenizing corpus")
        for _, seq in tokenization_log:
            tokens, bad_muts = self._tokenize(seq, tokens)
            if len(bad_muts) > 0:
                tokenization_log.set_postfix_str(f"WARNING: {len(bad_muts)} "
                                        "mutations could not be tokenized properly.")
        if len(bad_muts) > 0:
            print(f"\nBad Mutations: {bad_muts}")
        return tokens
    
    def _get_config(self):
        """
        Get the configuration of the tokenizer.

        Returns:
            dict: A dictionary containing the configuration of the tokenizer.
        """
        return {
            "tokenizer_class": self.__class__.__name__,
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "subtokenize": self.subtokenize, }
    
    @classmethod
    def _load_config(cls, tokenizer_config):
        """
        Load the tokenizer configuration from a dictionary json file.
        """
        return cls(**tokenizer_config)

    def save(self, save_path):
        """
        Save the tokenizer to a file.

        Args:
            path (str): The path to save the tokenizer.
        """
        if save_path is None:
            save_path = os.path.join(SAVE_TOKENIZER_PATH, 'manual_tokenizer.json')
        tokenizer_config = self._get_config()
        with open(save_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=4)

    @classmethod
    def load(cls, load_path):
        """
        Load the tokenizer from a file.

        Args:
            path (str): The path to load the tokenizer from.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Tokenizer file not found at {load_path}.")
        with open(load_path, 'r') as f:
            tokenizer_config = json.load(f)
        return cls._load_config(tokenizer_config)
    
########################## BPE-based VCF Tokenization ##########################
from transformers import PreTrainedTokenizer
class VCFTokenizer(PreTrainedTokenizer):
    """
    Loosely Inspired By
    https://github.com/dermatologist/genomic-tokenizer/blob/develop/genomic_tokenizer/core.py
    """
    def __init__(self, vocab=VOCAB_INIT, **kwargs):
        self.vocab = vocab
        self.id_to_token = {i: token for token, i in vocab.items()}
        self.token_to_id = vocab

        super().__init__()
        # super().__init__(
        #     eos_token=AddedToken("[SEP]", lstrip=False, rstrip=False),
        #     sep_token=AddedToken("[SEP]", lstrip=False, rstrip=False),
        #     cls_token=AddedToken("[CLS]", lstrip=False, rstrip=False),
        #     pad_token=AddedToken("[PAD]", lstrip=False, rstrip=False),
        #     mask_token=AddedToken("[MASK]", lstrip=False, rstrip=False),
        #     unk_token=AddedToken("[UNK]", lstrip=False, rstrip=False),
        #     add_prefix_space=False,
        #     **kwargs,)

        # self._extend_vocab(SPECIAL_TOKENS)
        self.pattern = SEQUENCE_PATTERN
        self.unk_pattern = UNK_PATTERN # for decoding to handle unk tokens

        self._allele_pattern = r"(?:\[UNK\]|(?:[ATCG]+(?:/[ATCG]+)*))"

    def __str__(self):
        tokenizer_str = "VCFTokenizer():\n"
        tokenizer_str += f"   Vocabulary Size: {self.vocab_size}\n"
        tokenizer_str += f"   Recognized Pattern: {self.pattern}\n"
        return tokenizer_str
    
    def get_vocab(self):
        """
        Get the vocabulary of the tokenizer.

        Returns:
            dict: A dictionary containing the vocabulary of the tokenizer.
        """
        return self.vocab
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def _extend_vocab(self, tokens):
        """
        Extend the vocabulary with new tokens.
        
        Args:
            tokens (list): A list of tokens to add to the vocabulary.
        """
        vocab_size = self.vocab_size
        for token in tokens:
            if (token not in self.vocab) and (not re.match(r"\d+", str(token))):
                # print(f"Adding token: {token} with id: {vocab_size}")
                self.vocab[token] = vocab_size
                vocab_size += 1

    def _update_tokenizer(self):
        """
        Update the vocabulary size and mappings.
        """
        self._update_total_vocab_size()
        self.id_to_token = {i: token for token, i in self.vocab.items()}
        self.token_to_id = self.vocab

    def _subtokenize_alleles(self, mutation):
        """
        Manually subtokenize a token in the form 'ref>alt' or 'a1|a2'.

        Args:
            mutation (str): A string representing a mutation.

        Returns:
            list: A list of tokens representing the mutation.
        """
        if '>' in mutation:
            # split by '>' for reference and alternative alleles
            split_by = '>'
        elif '|' in mutation:
            # split by '|' for phased sample alleles
            split_by = '|'
        elif '/' in mutation:
            # split by '/' for not phased genotype
            split_by = '/'
        else:
            raise ValueError(f"Invalid mutation format: {mutation}")
        allele_1, allele_2 = mutation.split(split_by)
        return allele_1, split_by, allele_2
    
    def _subtokenize_digits(self, mutation):
        """
        Manually subtokenize a long string of digits into individual tokens.
        EG: '24982470' -> ['2', '4', '9', '8', '2', '4', '7', '0'].

        Args:
            mutation (str): A string representing a mutation.

        Returns:
            list: A list of tokens representing the long number.
        """
        for digit in mutation:
            if digit not in self.vocab:
                self._extend_vocab([digit])
        return [digit for digit in mutation]

    def _tokenize(self, sequence, tokens = [], bad_muts = [], 
                  verbose = False, train = False):
        """
        Tokenize a sequence of mutations in the form 'chr:pos:ref>alt_a1|a2 
        chr:pos:ref>alt_a1|a2...'.
        """
        tokens = []
        for mut in sequence.split(' '): # get individual mutations
            if not re.fullmatch(self.pattern, mut):
                # if the mutation does not match the expected pattern, skip it
                # print(f"Skipping invalid mutation format: {mut}")
                continue
            else: 
                mut_split = mut.split(':')
                if '<' in mut_split[2] or len(mut_split) != 3: # invalid mutation
                    # print(f"Skipping invalid mutation format: {mut}")
                    mut = mut.split('_')[0]
                    if mut not in bad_muts:
                        bad_muts.append(mut)
                else:
                    chrom, pos, alleles = mut.split(':') # separate into chr, pos, ref>alt_a1|a2
                    tokens.extend(self._subtokenize_digits(chrom))
                    tokens.append(':') 
                    tokens.extend(self._subtokenize_digits(pos))
                    # tokens.extend([chrom, ':', pos])
                    # extend vocabulary for new tokens
                    if train:
                        self._extend_vocab([chrom, pos])
                    
                    mut_alleles, sample_gt = alleles.split('_') # split into ref>alt and a1|a2
                    # treat each allele in mutation specification and sample
                    # genotype as separate tokens
                    # EG: '1:545363:T>C_0|0' -> ['1', '545363', 'T', '>', 'C', '0', '0']
                    ref, _, alt = self._subtokenize_alleles(mut_alleles)
                    allele_1, split_by, allele_2 = self._subtokenize_alleles(sample_gt)
                    tokens.extend([ref, '>', alt, allele_1, split_by, allele_2])
                    if train:
                        self._extend_vocab([ref, alt, allele_1, split_by, allele_2])
        if verbose:
            print(f"Tokenized {len(tokens)} tokens from the sequence.")
            if len(bad_muts) > 0:
                print(f"Bad Mutations: {bad_muts}")
        return tokens
    
    def train(self, corpus=TRAIN_CORPUS):
        """
        Train the tokenizer on the provided corpus.

        This method tokenizes the corpus and extends the vocabulary with new tokens.
        """
        tokens = []
        corpus = load_corpus(corpus)
        # tokenize all sequences in the corpus
        tokenization_log = tqdm.tqdm(corpus.items(), 
                                     desc="Training tokenizer on corpus")
        for _, seq in tokenization_log:
            tokens = self._tokenize(sequence=seq, tokens=tokens, train=True)
        # Update 
        self._update_tokenizer()

        print(f"Tokenizer trained on corpus with {len(tokens)} tokens.")

    def encode(self, sequence):
        """
        Encode a sequence of mutations into token IDs.

        Args:
            sequence (str): A string representing a sequence of mutations.

        Returns:
            list: A list of token IDs representing the sequence.
        """
        tokens = self._tokenize(sequence)
        unk_id = self.token_to_id['[UNK]']
        token_ids = [self.token_to_id.get(token, unk_id) for token in tokens]
        return token_ids
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs into a sequence of mutations.

        Args:
            token_ids (list): A list of token IDs representing a sequence.

        Returns:
            str: A string representing the decoded sequence.
        """
        sequence = ''
        subsequence = ''
        gt_done = False
        for tok_id in token_ids:
            recon_token = self.id_to_token.get(tok_id, '[UNK]')
            if re.match(self._allele_pattern, recon_token):
                if not gt_done:
                    subsequence += ':' + recon_token
                else:
                    subsequence += recon_token + '_'
                gt_done = not gt_done
            elif recon_token == '[UNK]':
                subsequence += ':[UNK]_'
            else:
                subsequence += recon_token
            if re.fullmatch(self.unk_pattern, subsequence):
                sequence += subsequence + ' '
                subsequence = ''
            
        return sequence
    
    def _get_config(self):
        """
        Get the configuration of the tokenizer.

        Returns:
            dict: A dictionary containing the configuration of the tokenizer.
        """
        return {
            "tokenizer_class": self.__class__.__name__,
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "subtokenize": self.subtokenize, }
    
    @classmethod
    def _load_config(cls, tokenizer_config):
        """
        Load the tokenizer configuration from a dictionary json file.
        """
        return cls(**tokenizer_config)

    def save(self, save_path):
        """
        Save the tokenizer to a file.

        Args:
            path (str): The path to save the tokenizer.
        """
        if save_path is None:
            save_path = os.path.join(SAVE_TOKENIZER_PATH, 'VCFTokenizer_PTTF.json')
        tokenizer_config = self._get_config()
        with open(save_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=4)

    @classmethod
    def load(cls, load_path):
        """
        Load the tokenizer from a file.

        Args:
            path (str): The path to load the tokenizer from.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Tokenizer file not found at {load_path}.")
        with open(load_path, 'r') as f:
            tokenizer_config = json.load(f)
        return cls._load_config(tokenizer_config)