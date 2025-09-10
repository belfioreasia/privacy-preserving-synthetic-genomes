# import pysam
import os 
import re
import random
import torch
import tqdm
import json
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from datetime import datetime
import warnings


############################# GENERAL UTILS #############################
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_running_device(include_name = False):
    """
    Get the device to be used for PyTorch operations (CPU or GPU).
    """

    device_name = ""
    # check if CUDA is available
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        device = torch.device("cuda")
        device_name = "cuda"
    # check if MPS (Apple) is available
    elif torch.backends.mps.is_available():
        print("Using local MPS (Apple Silicon GPU)")
        device = torch.device("mps")
        device_name = "mps"
    else:
        warnings.warn("No GPUs available - running on CPU only")
        device = torch.device("cpu")
        device_name = "cpu"
        
    if include_name:
        return device, device_name
    else:
        return device

# global constants
EXTERNAL_DRIVE = '/Volumes/asia_2T/genomics' # path to the external storage 

DATASETS = ['1000GP', 'PGP', 'HAPNEST', 'TEST', 'RANDOM'] # for available datasets
RANDOM_DATASET = 'generated/random/random_vcf.vcf.gz'
TEST_DATASET = 'sources/test/input_files.vcf.gz'

CORPUS_DATAPATH = 'data/sources/json/corpus_chr22.json'
POPULATION_FILE = 'data/sources/1000GP/sample_lookup.csv'

# INITIAL default chromosome to use for genomic data training/generation 
DEFAULT_CHR = 22 # '1'

# from http://www.insilicase.com/Web/Chromlen.aspx
CHR_LENGTHS = {
    1: 249698942,
    2: 242508799,
    3: 198450956,
    4: 190424264,
    5: 181630948,
    6: 170805979,
    7: 159345973,
    8: 145138636,
    9: 138688728,
    10: 135186938,
    11: 133797422,
    12: 133275309,
    13: 114364328,
    14: 108136338,
    15: 102439437,
    16: 92211104,
    17: 83836422,
    18: 80373285,
    19: 58617616,
    20: 64444167,
    21: 46709983,
    22: 51857516, } # 51239281

# Tokenization rules for mutation sequences constructed using https://regexr.com
SEQUENCE_PATTERN = r"(?:[1-9]|1[0-9]|2[0-2]):\d+:(?:[ATCG]+(?:/[ATCG]+)*)>(?:[ATCG]+(?:/[ATCG]+)*)\_(?:[0-3][\/\|][0-3])"
MUTATION_PATTERN = r"(?:[1-9]|1[0-9]|2[0-2]):\d+:(?:[ATCG]+(?:/[ATCG]+)*)>(?:[ATCG]+(?:/[ATCG]+)*)\_(?:(?:[0-3][\/\|][1-3])|(?:[1-3][\/\|][0-3]))"

SNP_PATTERN = r"(?:[1-9]|1[0-9]|2[0-2]):\d+:(?:[ATCG]+(?:/[ATCG]+)*)>(?:[ATCG]+(?:/[ATCG]+)*)"
GT_PATTERN = r"(?:[0-3][\/\|][0-3])"

UNK_PATTERN = r"(?:[1-9]|1[0-9]|2[0-2]):\d+:(?:\[UNK\]|(?:[ATCG]+(?:/[ATCG]+)*))>(?:\[UNK\]|(?:[ATCG]+(?:/[ATCG]+)*))\_(?:[01][\/\|][01])"

############################# DATA STORAGE UTILS #############################
def get_data_dir(dataset='TEST', type='data', flags=''):
    """
    Returns the path of the desired file based on the provided flags for 
    easy retrieval of different datasets.
    NB: Due to the large size of genomic data, the sample files are stored 
    on an external drive.

    Available genomic datasets (sample, mutations and info files):
        - '1000GP': 1000 Genomes Project data
        - 'PGP': Personal Genomes Project data
        - 'HAPNEST': synthetic HAPNEST data
    
    A multi-sample (5000) vcf is available for testing purposes using
    the dataset parameter 'test'.
    A randomly generated multi-sample vcf is available using the dataset
    parameter 'random'.

    Args:
        dataset (str): name of the dataset to retrieve (defaults to 'TEST')
        type (str): (OPTIONAL) to specify the nature of the requested data: 
            - 'data': (default) to retieve the data/samples files
            - 'info': to retrieve the dataset info file
            - 'mutations': to retrieve the list of mutations in the dataset
        flags (str): (OPTIONAL) to specify the nature of the requested data:
            - 'L': to retrieve low-coverage data files
            - 'H': to retrieve high-coverage data files
    
    Returns:
        str: The file path of the desired data.
    """
    data_lookup = {
        'TEST': TEST_DATASET,
        'RANDOM': RANDOM_DATASET,
        '1000GP': os.path.join(EXTERNAL_DRIVE, 'reference/1000GP'),
        'PGP': os.path.join(EXTERNAL_DRIVE, 'reference/PGP'),
        # 'HAPNEST':  os.path.join(external_storage, 'intervene-synthetic/1mil/vcfs'),
        'HAPNEST':  os.path.join(EXTERNAL_DRIVE, 'synthetic/HAPNEST'),}

    info_lookup = {
        'TEST': None,
        'RANDOM': None,
        '1000GP': 'data/sources/1000GP/sample_info.xlsx',
        'PGP': 'data/sources/PGP',
        'HAPNEST': 'data/sources/HAPNEST',}

    flags_lookup = {
        'L': 'low-cov',
        'H': 'high-cov',
        '': '', } 

    dataset = dataset.upper()
    if dataset == '' or dataset is None:
        dataset = 'TEST'
    # Handle invalid dataset requests
    if dataset not in DATASETS and dataset != '':
        raise ValueError(f"Requested dataset '{dataset}' not found. \
                         Available datasets: {', '.join(DATASETS)}.")
    elif dataset == 'TEST' or dataset == 'RANDOM':
        return data_lookup[dataset]
    else:
        if flags not in flags_lookup:
            raise ValueError(f"Invalid requested flag '{flags}'. \
                             Must be any of: {', '.join(flags_lookup.keys())}.")
        
        if (type == 'data') or (type == ''):
            # If no type is specified, return the VCF data path
            return os.path.join(data_lookup[dataset], flags_lookup[flags])
        elif type == 'info':
            return info_lookup[dataset]
        elif type == 'mutations':
            return os.path.join(data_lookup[dataset], 'mutations')
        else:
            raise ValueError(f"Invalid file request: '{type}'. Must be\
                              any of: 'data', 'info', 'mutations.")

def get_VCF_file(dataset='TEST', sample_ids=None, chr=DEFAULT_CHR, flags='H', zip=True):
    """
    Returns the chosen VCF subset of samples data file for a specified dataset.
    If no sample ID is specified, returns the path to the full dataset directory.

    A small (5000) multi-sample vcf file for testing purposes is available using
    the dataset parameter 'test'.

    Args:
        dataset (str) : name of the dataset to retrieve (defaults to 'TEST')
        sample_id (str, list): ID of the sample(s) to retrieve. Can be a single 
                                 ID or a list of IDs. If None, returns the path 
                                 to the full requested dataset.
        flags (str): (OPTIONAL) to specify the nature of the requested data:
            - 'L': (default) to retrieve low-coverage data files
            - 'H': to retrieve high-coverage data files
        zip (bool): (OPTIONAL) to return the gzipped version of the 
                      requested file (if available) (defaults to True)

    Returns:
        str: The file path of the sample(s) data.
    """    
    # ignore X and Y chromosomes for compatibility with HAPNEST data
    if not (dataset.endswith('.vcf') or dataset.endswith('.vcf.gz')):
        assert isinstance(chr, int) and (0 < chr < 23), "Chromosome must be an integer between 1 and 22."
        
        file_dir = get_data_dir(dataset=dataset, type='data', flags=flags)
        file_type = '.vcf.gz' if zip else '.vcf'

        # multi sample vcfs filenames as taken from original downloaded site files
        data_filenames = {
            '1000GP': f'ALL.chr{chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased' if chr <= 6 \
                        else f'ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes',
            'PGP': f'ALL.chr{chr}.GRCh38',
            'HAPNEST': f'synth.chr{chr}',}
        
        if dataset.upper() == 'TEST':
            filepath = file_dir
        else:
            filepath = os.path.join(file_dir, f'{data_filenames[dataset.upper()]+file_type}')
    else:
        filepath = dataset
    assert os.path.exists(filepath), f"VCF file '{filepath}' does not exist." 

    if (sample_ids is None) or (sample_ids == '') or (sample_ids == []) or (sample_ids == 'all'):
        # If no sample ID is specified, return file for ALL samples
        # return pysam.VariantFile(filepath)
        return VCF(filepath, gts012=True)
    else:
        # WARNING: pysam.VariantFile().subset_samples() not properly working?
        # return pysam.VariantFile(filepath).subset_samples([sample for sample in sample_ids])
        return VCF(filepath, gts012=True, samples=sample_ids)

def get_data(dataset='TEST', sample_ids=None, mutations=None, include_alleles=False,
             chr=DEFAULT_CHR, flags='H', zip=True, save=False, save_path=None):
    """
    Returns a dataset of genotypes for a chosen subset of samples from a specified
    dataset. If no sample ID is specified, returns the data for the full dataset.
    If no mutations are specified, returns all mutations in the dataset. 
    NB: desired mutations can only be within the specified chromosome.

    The data is returned in the format of a dictionary with sample IDs as keys
    and dictionaries of mutations as keys and their genotypes as values.
    The formatted data can be saved to a file if the save parameter is set to True.

    A small (5000) multi-sample vcf file for testing purposes is available using
    the dataset parameter 'test'.

    Args:
        dataset (str): name of the dataset to retrieve (defaults to 'TEST')
        sample_ids (str, list): ID of the sample(s) to retrieve. Can be a single 
                                ID or a list of IDs. If None, returns the data for
                                the full requested dataset.
        mutations (str, list): List of mutation IDs to retrieve. If None or 'all',
                               retrieves all mutations in the dataset.
        chr (str): (OPTIONAL) chromosome to retrieve data for (defaults to 'chr1')
        flags (str): (OPTIONAL) to specify the nature of the requested data:
            - 'L': (default) to retrieve low-coverage data files
            - 'H': to retrieve high-coverage data files
        zip (bool): (OPTIONAL) to return the gzipped version of the 
                    requested file (if available) (defaults to True)
        save (bool): (OPTIONAL) to save the formatted data to a file (defaults to False)
        save_path (str): (OPTIONAL) path to save the formatted data file.
                        If None, saves to the default path based on the dataset name.
    
    Returns:
        pd.DataFrame: A DataFrame with sample IDs as rows and mutations as columns,
                      containing the genotypes for each sample and mutation.
    """
    if isinstance(dataset, VCF):
        vcf_file = dataset
    else:
        vcf_file = get_VCF_file(dataset=dataset, sample_ids=sample_ids, 
                                chr=chr, flags=flags, zip=zip)
    genotypes = get_samples_genotypes(vcf_file=vcf_file,
                                      dataset=dataset,
                                      sample_ids=sample_ids, 
                                      mutations=mutations,
                                      include_alleles=include_alleles)
    if sample_ids is None or sample_ids == [] or sample_ids == '':
        sample_ids = list(genotypes.keys())
    found_mutations = list(genotypes[sample_ids[0]].keys())
    genotypes_df = pd.DataFrame.from_dict({sample: [str(genotypes[sample][mut]) 
                                            for mut in found_mutations] 
                                            for sample in genotypes.keys() },
                                            orient='index', columns=found_mutations)
    # genotypes_df.index.name = 'sample_id'
    # genotypes_df.reset_index(inplace=True)
    if save:
        if save_path is None:
            save_path = os.path.join(get_data_dir(dataset=dataset, type='data', flags=flags),
                                     f"{dataset}_genotypes_chr{chr}.json")
        with open(save_path, 'w') as f:
            json.dump(genotypes_df, f, sort_keys=True, indent=4)
    print(f"Loaded {len(genotypes_df)} samples with {len(genotypes_df.columns)} mutations.")
    return genotypes_df

def get_partial_mutation_data(data, num_hidden=0.3, return_hidden=False):
    """ 
    Hide a random subset of mutations in the dataset for each sample. 
    By default, 30% of SNPs are hidden, chosen at random.

    Args:
        data (pd.Dataframe): Genotype data passed as sample-by-mutation matrix, with
                                sample genotypes as entries and sample ids as indices.
        hide_mutations (float): Proportion of mutations to hide (defaults to 30%).
        return_hidden (bool): to return or not the complementary data subset with
                                genotypes for the hidden mutations (defaults to False)

    Returns:
        tuple: two DataFrame with the same structure as the input data
                as non-overlapping subsets of the input mutations for 
                all samples in the input dataset.
    """
    all_mutations = data.columns.tolist()
    num_to_keep = int(len(all_mutations)*(1-num_hidden))
    keep_mutations = random.sample(population=all_mutations, k=num_to_keep)
    if return_hidden:
        hidden_mutations = [mut for mut in all_mutations if mut not in keep_mutations]
        assert all(mut not in keep_mutations for mut in hidden_mutations), \
                'Kept and Hidden Mutation Subsets Overlap.'
        return data[keep_mutations], data[hidden_mutations]
    else:
        return data[keep_mutations], None

def get_partial_samples_dataset(data, samples_to_hide, return_hidden=False):
    """ 
    Hide a random subset of samples in the dataset. Can be used to get individual
    samples for testing purposes, or to simulate a scenario where only a subset of
    samples is available for analysis. 

    Args:
        data (pd.Dataframe): Genotype data passed as sample-by-mutation matrix, with
                                sample genotypes as entries and sample ids as indices.
                                with sample genotypes as entries.
        samples_to_hide (str, list, int, float): which samples to hide. Can either be
            - a list of sample IDs or a single sample ID to hide
            - an integer representing the number of samples to hide at random
            - a float representing the proportion of samples to hide at random
        return_hidden (bool): to return or not the complementary data subset with
                            genotypes of the hidden samples (defaults to False)

    Returns:
        tuple: two DataFrame with the same structure as the input data
                as non-overlapping subsets of the input samples.
    """
    all_samples = data.index.tolist()
    if isinstance(samples_to_hide, str):
            samples_to_hide = [samples_to_hide]
    if isinstance(samples_to_hide, list):
        assert all(sample in all_samples for sample in samples_to_hide), \
                'Unknown Requested Sample not found in VCF.'
        keep_samples = [sample for sample in all_samples if sample not in samples_to_hide]
    else:
        if isinstance(samples_to_hide, (int)):
            num_to_keep = len(all_samples) - samples_to_hide
        elif isinstance(samples_to_hide, (float)):
            num_to_keep = int(len(data)*(1 - samples_to_hide))
        # print(f"Keeping {num_to_keep} samples out of {len(all_samples)} total samples.")
        keep_samples = random.sample(population=all_samples, k=num_to_keep)
    if return_hidden:
        hidden_samples = [sample for sample in all_samples if sample not in keep_samples]
        assert all(sample not in keep_samples for sample in hidden_samples), \
                'Kept and Hidden Samples Subsets Overlap.'
        return data.loc[keep_samples], data.loc[hidden_samples]
    else:
        return data.loc[keep_samples], None

def get_sequences_from_file(gt_file):
    """
    Load sequences from a ground truth file.
    :param gt_file: Path to the ground truth file.
    :return: List of sequences.
    """
    # Read the ground truth file
    with open(gt_file, 'r') as f:
        if gt_file.endswith('.json'):
            sequences = json.load(f)
            sequences = list(sequences.values())
        if gt_file.endswith('.txt'):
            sequences = f.readlines()
    return sequences

def text_to_json(sequences, save_path):
    """
    Convert a text file of sequences to a JSON file.
    :param text_file: Path to the text file containing sequences.
    :return: Dictionary of sequences.
    """
    assert save_path.endswith('.json'), "Save path must be a .json filepath."
    gt_dict = {f'synth_{i}': seq.strip('\n') for i, seq in enumerate(sequences)}
    if save_path is not None:
        with open(save_path, 'x') as f:
            json.dump(gt_dict, f, sort_keys=False, indent=4)
    return gt_dict

def load_corpus(VCF_corpus=CORPUS_DATAPATH):
        """
        Creates a training corpus from the VCF dataset as a list of genotype 
        sequences for each sample.

        Sample json file corpus:
        {   sample_1: "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_0|1",    
            sample_2: "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_1|1",    
            sample_3: "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_0|1"   }

        Args:
            VCF_dataset (str or json file): (path to the) json file containing the 
                                            pre-processed VCF data.
        """
        assert isinstance(VCF_corpus, str) and VCF_corpus.endswith('.json'), \
            "VCF_corpus must be a path to a JSON file containing the corpus data."

        print(f"Loading corpus from {VCF_corpus}...")
        with open(VCF_corpus) as json_file:
            corpus = json.load(json_file)

        return corpus # returns dict
    
def corpus_to_json(corpus=CORPUS_DATAPATH, population_file=POPULATION_FILE, labels=False,
                   has_sample_names=False, include_sample_names=False,
                   save_path=None):
    """
    """

    genotypes = get_sequences_from_file(corpus)
    if isinstance(genotypes[0], dict):
        genotypes = list(genotypes.values())
    valid_gts = [gt.split(' ') for gt in genotypes]

    if labels:
        populations = pd.read_csv(population_file, sep="\t")
    
    if has_sample_names and include_sample_names:
        sample_ids = list(gt[0] for gt in valid_gts)
        start_id = 1
    else:
        if sample_ids == []:
            sample_ids = list(f'sample_{i}' for i in range(len(valid_gts)))
        start_id = 0
    # print(sample_ids[:10]) 

    serialised_dataset = {id: {
        'genotype': ' '.join(valid_gts[i][start_id:]),
        'population': populations[populations['Sample'] == id]['Population'].values[0] if labels else None
        } for i, id in enumerate(sample_ids)}

    if save_path:
        with open(save_path, 'w+') as f:
            json.dump(serialised_dataset, f, sort_keys=False, indent=4)

    return serialised_dataset

############################## VCF FILE UTILS ##############################
def format_mutation(mutation, include_alleles=False):
    """
    Standardise mutations into the format 'chromosome:position:ref:alt'.
    Supported conversion from the following formats:
        - chromosome:position or chromosome:position:REF:ALT
        - (chromosome, position) or (chromosome, position, REF, ALT)
    with chromosome as "chrX" (UCSC), "NC_00000X.11" (NCBI), or "X" (EBI)

    Args:
        mutation (str, tuple): mutation in the format 'chromosome:position' or
                               as a tuple (chromosome, position)

    Returns:
        tuple (int, int): chromosome number and position
    """
    if isinstance(mutation, str):
        mutation = mutation.split(':')
    elif not isinstance(mutation, (tuple, list, pd.Series)):
        raise TypeError("Format Error: Mutation must be a str in the format \
                        'chromosome:position' or a tuple (chromosome, position).")

    if isinstance(mutation, pd.Series):
        chr, pos, ref, alt = mutation['CHR'], mutation['POS'], mutation['REF'], mutation['ALT']
    elif len(mutation) > 2:
        chr, pos, ref, alt = mutation[0], mutation[1], mutation[2], mutation[3]
    elif len(mutation) == 2:
        chr, pos = mutation[0], mutation[1]
    else:
        raise ValueError("Format Error: Mutation must be in the format 'chromosome:position'\
                         or 'chromosome:position:ref:alt'.")
    
    chr = str(chr).lower() # handle case insensitivity
    chr = chr.removeprefix('chr') # only keep chromosome number
    if include_alleles:
        return (int(chr), int(pos), ref, alt)
    else:
        return (int(chr),int(pos))

def standardise_mutations(mutations=None, include_alleles=False):
    """
    Standardise a list of mutations into the format 'chromosome':position'.
    Supported conversion from the following formats:
        - chromosome:position (E.g. 'chr1:123456' or '1:123456')
        - (chromosome, position) (E.g. ('chr1', 123456) or ('1', 123456))

    Args:
        mutations (list): List of mutations in the format 'chromosome:position'
                          or as tuples (chromosome, position).

    Returns:
        list: tuples of mutation chromosome numbers and positions.
    """

    assert mutations is not None, "No genetic mutation received."
    if isinstance(mutations, (tuple, str)):
        mutations = [mutations]
    
    if isinstance(mutations, pd.DataFrame):
        formatted_mutations = mutations.apply(format_mutation, axis=1).tolist()
    else:
        formatted_mutations = []
        for mut in mutations:
            formatted_mutations.append(format_mutation(mut, include_alleles))
    return formatted_mutations

def get_mutations(filepath=None, dataset='TEST', chr=DEFAULT_CHR, dataframe=False):
    """
    Returns the path to the mutations file for a specified dataset and chromosome.
    The mutations file contains a list of mutations in the format 'chromosome:position'.

    Args:
        dataset (str): name of the dataset to retrieve (defaults to 'TEST')
        chr (int): chromosome number to retrieve mutations for (defaults to DEFAULT_CHR)
        flags (str): (OPTIONAL) to specify the nature of the requested data:
            - 'L': (default) to retrieve low-coverage data files
            - 'H': to retrieve high-coverage data files

    Returns:
        list or pandas.Dataframe: containing the mutations for the specified chromosome.
    """
    assert isinstance(chr, int) and (0 < chr < 23),\
          "Chromosome must be an integer between 1 and 22."
    if filepath is None:
        mutations_file = os.path.join(get_data_dir(dataset=dataset, type='mutations'),
                        f'chr{chr}_snp.txt')
    else:
        mutations_file = filepath
    if not os.path.exists(mutations_file):
        raise FileNotFoundError(f"Mutations file for dataset '{dataset}' and chromosome \
                                 {chr} does not exist.")
    mutations = pd.read_csv(mutations_file, sep=':', header=None,
                            names=['CHR', 'POS', 'REF', 'ALT'])
    if dataframe:
        return mutations
    else:
        return mutations.apply(lambda row: format_mutation(row, include_alleles=False), 
                               axis=1).tolist()

def get_samples_genotypes(vcf_file, mutations_file=None, dataset='TEST', sample_ids='all', mutations='all', chr=DEFAULT_CHR,
                          include_alleles=True, encode_freq=False, sequential=False):
    """
    Get the genotypes of a list of specified mutations for one or more samples in 
    the VCF file. Samples and mutations can be randomly selected or specified by ID.
    If no sample IDs are specified, genotypes for 50 random samples are returned.
    If no mutations are specified, all mutations in the VCF file are returned.

    Sample input VCF (.vcf.gz):
            #CRHOM      #POS      #REF    #ALT    sample_1    sample_2  ...
            chr1    1   545363	    T	    C	    0|0         0|1     ...
            chr1    1   793571	    A	    G	    0|1         1|1     ...
    ...

    Sample dictionary returned:
    A) if sequential=False (nested dictionary):
        {   
            sample_1: { 
                        '1:545363_T>C': '0|0',
                        '1:793571_A>G': '0|1',
                        ...
                      }    
            sample_2: { 
                        '1:545363_T>C': '0|1',
                        '1:793571_A>G': '1|1',
                        ...
                      }
            ... 
        }
    B) if sequential=True (dictionary of strings):
        {   
            sample_1: "1:545363_T>C_0|0 1:793571_A>G_0|1 ...",    
            sample_2: "1:545363_T>C_0|0 1:793571_A>G_1|1 ...",  
            ... 
        }
    
    Args:
        vcf_file (str): cyvcf2 VCF object (Open VCF file).
        dataset (str): name of the dataset to retrieve (defaults to 'TEST')
        sample_ids (str, int, list): List of sample IDs to retrieve. If 'all',
                                retrieves all samples in the VCF file. 
                                If int, retrieves that many random samples.
                                If None, retrieves 50 random samples.
        mutations (str, int, list): List of mutation IDs to retrieve. If None or 
                                    'all' retrieves all mutations in the VCF file.
                                    If int, retrieves that many random mutations.
        chr (int): chromosome number to retrieve data for (defaults to DEFAULT_CHR)
        include_alleles (bool): (OPTIONAL) to include the reference and alternate
                                alleles in the returned tuple (defaults to False)
        encode_freq (bool): (OPTIONAL) to encode the genotype as a frequency (0, 1,
                            or 2) instead of the allele values (0|0, 0|1, 1|0, 1|1)
        sequential (bool): (OPTIONAL) to return the genotypes as sequential string of
                        mutations and genotypes for each sample (defaults to False)
    
    Returns:
        dict: A dictionary with sample IDs as keys and dictionaries of 
              mutations as keys and their genotypes as values.
              Example: {sample_id: {mutation_id: (allele_1, allele_2)}}
    """
    available_samples = vcf_file.samples

    print(len(available_samples), "available samples in the VCF file.")

    if mutations is not None and not isinstance(mutations, (int, str, tuple, list)):
        raise TypeError("Mutations must be a str or list of mutation IDs or None.")
    
    # Specify (number of) samples to include (randomly or by ID)
    if sample_ids is None or sample_ids == []:
        # return 50 random samples if no specific sample IDs are provided
        sample_ids = 50
    elif isinstance(sample_ids, str):
        if sample_ids == 'all':
            sample_ids = available_samples
        else:
            sample_ids = [sample_ids] 
    elif not isinstance(sample_ids, (list, int)):
        raise TypeError("Sample IDs must be an int, str, list, or None.")
    if isinstance(sample_ids, int):
        # extract random samples from available ones in VCF file
        sample_ids = random.choices(population=available_samples, k=sample_ids)
    
    # handle requested sample IDs not in VCF file
    if not all(sample in available_samples for sample in sample_ids):
        passed_samples = len(sample_ids)
        sample_ids = [sample for sample in sample_ids if sample in available_samples]
        warnings.warn(f"{passed_samples-len(sample_ids)} sample IDs not found in the VCF file.")
    sample_indices = [available_samples.index(sample) for sample in sample_ids]
    
    if isinstance(mutations, (str, tuple, list)) and mutations != 'all':
        mutations = standardise_mutations(mutations)
    elif isinstance(mutations, int):
        # get a random subset of mutations for the specified chromosome
        # NB: for now only supports single chromosome mutations
        if (not isinstance(chr, int)) or (chr < 1 or chr > 22):
            warnings.warn(f"Invalid chromosome {chr} specified. Using default chromosome {DEFAULT_CHR}.")
            chr = DEFAULT_CHR
        all_mutations = get_mutations(filepath=mutations_file, dataset=dataset, chr=chr, dataframe=False)
        # all_mutations = standardise_mutations(all_mutations, include_alleles=include_alleles)
        mutations = random.sample(population=all_mutations, k=mutations)

    genotypes_per_sample = {sample:{} for sample in sample_ids} if not sequential \
                            else {sample:'' for sample in sample_ids}
    found_snps = 0
    bad_samples = []

    for rec in tqdm.tqdm(vcf_file, desc="Processing VCF records"): # iterate through each mutation 
        if mutations == 'all' or (int(rec.CHROM), int(rec.POS)) in mutations:
            if not (any('<' in ref for ref in rec.REF) or any('<' in alt for alt in rec.ALT)):
                found_snps += 1
                if include_alleles:
                    alt_allele = '/'.join(alt for alt in rec.ALT) # handle multiple alternate alleles
                                                                # EG. ['C','T', 'A'] -> 'C/T/A'
                    mutation_key = f"{rec.CHROM}:{rec.POS}:{rec.REF}>{alt_allele}"
                else:
                    mutation_key = f"{rec.CHROM}:{rec.POS}"
                # print(f"Processing {mutation_key}")
                all_genotypes = rec.genotypes  # genotypes for the current mutation for ALL samples
                genotype_quals = rec.gt_types
                for i,sample in enumerate(sample_ids):
                    if genotype_quals[sample_indices[i]] == 3: # UNKOWN genotype = BAD read (IGNORE)
                        warnings.warn(f"Sample {sample} has an UNKNOWN genotype for mutation {mutation_key}.")
                        bad_samples.append((sample_indices[i], sample))
                        if sequential:
                            genotypes_per_sample[sample] += f' {mutation_key}_[UNK] '
                        else:
                            genotypes_per_sample[sample][mutation_key] = '[UNK]'
                    else:
                        if encode_freq:
                            if sequential:
                                genotypes_per_sample[sample] += f' {mutation_key}_{genotype_quals[sample_indices[i]]} '
                            else:
                                genotypes_per_sample[sample][mutation_key] = genotype_quals[sample_indices[i]]
                        else:
                            # already checked that sample is in vcf_file.samples
                            allele_1, allele_2, phased = all_genotypes[sample_indices[i]]   # GT [[allele_1, allele_2, phased_bool]...]
                                                                                            # Eg. 2 samples of 0/0 and 1|1 would be 
                                                                                            #     [[0, 0, False], [1, 1, True]]
                            if sequential:
                                genotypes_per_sample[sample] += f' {mutation_key}_{allele_1}|{allele_2}' if phased \
                                                                        else f' {mutation_key}_{allele_1}/{allele_2}'
                            else:
                                genotypes_per_sample[sample][mutation_key] = f'{allele_1}|{allele_2}' if phased \
                                                                            else f'{allele_1}/{allele_2}'
        if mutations != 'all' and found_snps == len(mutations):
            break # exit if all mutations have been analysed
    if found_snps == 0:
        print("No match found in the VCF file for the specified mutations.")
    else:
        print(f"Processed {found_snps} mutations for {len(genotypes_per_sample)} samples.")
    if len(bad_samples) > 0:
        warnings.warn(f"{len(bad_samples)} samples had UNKNOWN genotypes for some mutations: {bad_samples}")
    return genotypes_per_sample

def vcf_to_sequential(vcf_file, dataset, mutations_file=None, sample_ids=10, mutations='all', chr=DEFAULT_CHR,
                      flags='H', save=False, save_path=None):
        """
        Creates a training corpus from the VCF dataset as a list of genotype sequences
        for each sample.

        Sample input VCF (.vcf.gz):
        	            #CRHOM      #POS      #REF    #ALT    sample_1    sample_2  ...
        chr1:545363	       1       545363	    T	    C	    0|0         0|1     ...
        chr1:793571	       1       793571	    C	    T	    0|1         1|1     ...
        ...

        Sample sample-wise dictionary corpus:
        {   sample_1: "1:545363_T>C_0|0 1:793571_C>T_0|1 ...",    
            sample_2: "1:545363_T>C_0|0 1:793571_C>T_1|1 ...",  }

        Args:
            vcf_file (cyvcf2.VCF): Open VCF file object.
            dataset (str): Name of the dataset to retrieve.
            sample_ids (int, list): Specified samples to retrieve as a list of sample IDs
                                    or number of samples to retrieve at random.
            mutations (str, list): List of mutation IDs to retrieve. If None or 'all'
                                   retrieves all mutations in the VCF file.
            chr (int): Chromosome number to retrieve data for (defaults to DEFAULT_CHR).
            flags (str): (OPTIONAL) to specify the nature of the requested data:
                        - 'L': (default) to retrieve low-coverage data files
                        - 'H': to retrieve high-coverage data files
            save (bool): (OPTIONAL) to save the formatted corpus to a .json file
            save_path (str): (OPTIONAL) path to save the formatted corpus file to. If None, 
                             saves to the default path based on the dataset and chromosome.
        Returns:
            dict: A dictionary with sample IDs as keys and mutation info with genotypes
                    as values.
        """
        if not isinstance(vcf_file, VCF):
            raise TypeError("Invalid Parameter: dataset must be a file path or DataFrame.")

        corpus_dict = get_samples_genotypes(vcf_file, mutations_file=mutations_file, dataset=dataset, sample_ids=sample_ids, mutations=mutations,
                                            chr=chr, include_alleles=True, sequential=True)
        
        if save:
            if save_path is None or save_path == '':
                save_path = os.path.join(get_data_dir(dataset=dataset, type='data', flags=flags),
                                        f"{dataset}_corpus_chr{chr}.json")
                
            # save on file
            with open(save_path, 'w+') as f:
                json.dump(corpus_dict, f, sort_keys=True, indent=4)

        return corpus_dict

def corpus_to_VCF(corpus, imputation_method, save_path=None, **kwargs):
    """
    Converts a corpus of sample-wise mutations into a VCF file format.
    
    The input corpus mutations are expected to be in the following format: 
    
                            "CHROM:POS:REFs>ALTs_GT" 
                    (EG. "1:545363_T>C_0|0 1:793571_A>G_0|1 ...")

        #CHROM  POS ID  REF ALT QUAL    FILTER  INFO    FORMAT  synth_1 synth_2
        1   545363  .  T  C  .  PASS   NS=2;AC=1;AF=0.25;AN=4    GT   0|0  0|1
        1   793571  .  A  G  .  PASS   NS=2;AC=1;AF=0.75;AN=4    GT   1|1  0|1

    Args:
        corpus (dict, list, str): per-sample genotypes for mutations. Can be:

            - a dictionary with sample IDs as keys and mutation info with genotypes as values 
                    EG. {sample_1: "1:545363_T>C_0|0 1:793571_A>G_0|1 ...",
                        sample_2: "1:545363_T>C_0|0 1:793571_A>G_1|1 ..."}
            - a list of strings with mutations for each sample
                    EG. ["1:545363_T>C_0|0 1:793571_A>G_0|1 ...",
                        "1:545363_T>C_0|0 1:793571_A>G_1|1 ..."]
            - a string with mutations for a single sample
                    EG. "1:545363_T>C_0|0 1:793571_A>G_0|1 ..."
            - a string path to a .json file with the corpus data.
        
        kwargs (dict): A dictionary with additional parameters for VCF conversion.
        imputation_method (str): The (generative) model used for data imputation.
        save (bool): (OPTIONAL) to save the formatted corpus to a .json file
        save_path (str): (OPTIONAL) path to save the formatted corpus file to. If None, 
                         saves to the default path based on the dataset and chromosome.

    Returns:
        str: A string representation of the VCF file.
    """
    verbose = kwargs.get('verbose', True)
    if isinstance(corpus, str):
        if corpus.endswith('.json'):
            # load corpus from json file
            with open(corpus) as json_file:
                corpus = json.load(json_file)
        else:
            # assume corpus is a single sequence
            corpus = {'synth_1': corpus}
    elif isinstance(corpus, list):
        # convert list to dict with sample IDs as keys
        if isinstance(corpus[0], list):
            corpus = {f'synth_{i+1}': ' '.join(muts) for i, muts in enumerate(corpus)}
        else:
            corpus = {f'synth_{i+1}': muts for i, muts in enumerate(corpus)}
    elif not isinstance(corpus, dict):
        raise TypeError("Input corpus not .json file or path to json file.")
    
    allowed_chroms = kwargs.get('allowed_chroms', list(range(1,23)))
    if isinstance(allowed_chroms, int):
        allowed_chroms = [allowed_chroms]
    
    # 1. Gather VCF Header fields
    header_fields = ['CHROM', 'POS', 'ID', 'REF', 'ALT']
    include_metrics = {}

    include_filter = kwargs.get('include_filter', True)
    include_info = kwargs.get('include_info', True)

    if include_filter:
        header_fields.append('QUAL')
        header_fields.append('FILTER')
        include_metrics['FILTER'] = []
    if include_info:
        header_fields.append('INFO')
        include_metrics['INFO'] = {}
        include_metrics['INFO']['NS'] = []  # number of samples with data
        include_metrics['INFO']['AC'] = []  # ALT allele count in genotypes
        include_metrics['INFO']['AF'] = []  # allele frequency for each ALT allele
        include_metrics['INFO']['AN'] = []  # total alleles in called genotypes (2*samples)
    
    header_fields.append('FORMAT')

    # 2. Gather Mutation data (row-wise) and stats
    samples_list = corpus.keys()
    
    # chrom = set()
    chrom = {"ERROR"}
    corpus_genotypes = {}
    if verbose:
        vcf_log = tqdm.tqdm(corpus.items(), 
                            desc="Gathering mutation information from corpus")  
    else:
        vcf_log = corpus.items()

    bad_muts = set()
    for sample_id, genotypes in vcf_log:
        if isinstance(genotypes, str):
            mutations = re.findall(SEQUENCE_PATTERN, genotypes)
        else:
            mutations = genotypes
        for mut in mutations:
            # print(f"Processing mutation {mut} for sample {sample_id}")
            chr, pos, alleles = mut.split(":")

            if int(chr) not in allowed_chroms:
                continue

            if chr not in chrom:
                if len(chrom) <= 1 and "ERROR" in chrom:
                    chrom.remove("ERROR")
                    chrom = {int(chr)}
                else:
                    chrom.add(int(chr))

            ref_alt, gt = alleles.split("_")
            if '<' in ref_alt:
                continue  # skip bad read variants
            else:
                ref, alt = ref_alt.split(">")
                if int(pos) > CHR_LENGTHS[int(chr)]:
                    bad_muts.add(f'{chr}:{pos}:{ref}:{alt}')
                    continue
                if (chr, pos, ref, alt) not in corpus_genotypes.keys():
                    corpus_genotypes[(chr, pos, ref, alt)] = {}
                corpus_genotypes[(chr, pos, ref, alt)][sample_id] = gt 

    if verbose:
        print(f"Found {len(corpus_genotypes)} unique mutations in {len(chrom)} chromosomes.")
    
    # 3. Create VCF Body
    vcf_body = ""
    filter_fields = set()
    if verbose:
        vcf_log = tqdm.tqdm(enumerate(corpus_genotypes.items()),
                            desc="Writing mutations to VCF")
    else:
        vcf_log = enumerate(corpus_genotypes.items())
    for mut_num, ((chr, pos, ref, alt), gts) in vcf_log:
        vcf_body += f"{chr}\t{pos}\tchr{chr}:{pos}\t{ref}\t{alt}\t" 
        
        samples_matched = [False for _ in samples_list]
        samples_gts = ''
        alt_allele_nums = len(alt.split('/'))

        # initialize fields for included stats
        if include_filter:
            if len(include_metrics['FILTER']) < mut_num + 1:
                include_metrics['FILTER'].append('PASS')
        if include_info:
            if len(include_metrics['INFO']['NS']) < mut_num + 1:
                include_metrics['INFO']['NS'].append(0)
                include_metrics['INFO']['AC'].append([0 for _ in range(alt_allele_nums)])
                include_metrics['INFO']['AF'].append([0 for _ in range(alt_allele_nums)])
                include_metrics['INFO']['AN'].append(0)

        for i,sample in enumerate(samples_list):
            for sample_in_gt, gt in list(gts.items()):
                if sample_in_gt == sample:
                    # print("Found", sample, sample_in_gt)
                    samples_gts += f"{gt}\t"
                    samples_matched[i] = True
                    if include_info:
                        include_metrics['INFO']['NS'][mut_num] += 1
                        # TODO: add support for multiallelic mutations
                        for allele_num in range(alt_allele_nums):
                            if gt in [f'{allele_num}|{allele_num}', 
                                      f'{allele_num}/{allele_num}']:
                                # count homozygous alt mutations (2 alt alleles)
                                include_metrics['INFO']['AC'][mut_num][allele_num] += 2
                            elif gt in [f'0|{allele_num}', f'0/{allele_num}', 
                                        f'{allele_num}|0', f'{allele_num}/0']:
                                # count heterozygous alt mutations (1 alt allele)
                                include_metrics['INFO']['AC'][mut_num][allele_num] += 1
                    break
            # default genotype for samples not in the corpus
            if not samples_matched[i]:
                # print("Default", sample, sample_in_gt)
                # missing sample genotype for current mutation
                samples_gts += "0|0\t"
            if include_info:
                for allele_num in range(alt_allele_nums):
                    include_metrics['INFO']['AN'][mut_num] += 2
                    include_metrics['INFO']['AF'][mut_num][allele_num] = include_metrics['INFO']['AC'][mut_num][allele_num] / include_metrics['INFO']['AN'][mut_num]
        
        if include_filter:
            s_filter = include_metrics['INFO']['NS'][mut_num] / len(samples_list)                                                 
            if s_filter < 0.5:
                # keep PASS if at least 50% of samples have data, otherwise sXX
                filter_fields.add(int(s_filter*100))
                include_metrics['FILTER'][mut_num] = f's{int(s_filter*100)}'
            vcf_body += '.\t'  # QUAL - quality score (NB: not computed here)
            vcf_body += f"{include_metrics['FILTER'][mut_num]}\t"
        if include_info: # '.'
            for field, value in include_metrics['INFO'].items():
                vcf_body += f"{field}=" 
                if isinstance(value[mut_num], list):
                    # handle multiallelic mutations
                    vcf_body += ','.join([str(v) for v in value[mut_num]])
                else:
                    vcf_body += str(value[mut_num])
                vcf_body += ";"
            vcf_body += "\t"
        vcf_body += "GT\t"
        vcf_body += f"{samples_gts}\n"

    # 4. Create VCF Header
    vcf_header = "##fileformat=VCFv4.1\n" \
                f"##fileDate={datetime.today().strftime('%Y%m%d')}\n" \
                f"##source={imputation_method}\n" \
                # "##reference=file:///seq/references/1000GenomesPilot-NCBI36.fasta\n"
    for chr in sorted(chrom):
        if chr != "ERROR":
            vcf_header += f"##contig=<ID={chr},length={CHR_LENGTHS[int(chr)]},assembly=GRCh38>\n"
                # "##contig=<ID={chr},length={CHR_LENGTHS[chr]},assembly=GRCh38,md5=f126cdf8a6e0c7f379d618ff66beb2da,species=\"Homo sapiens\",taxonomy=x>\n"
        else:
            warnings.warn(f"Error in chromosome read ({chr}).")

    if include_filter:
        # sample_data = set(metric/len(samples_list) for metric in include_metrics['INFO']['NS'])
        # FILTER - filter status: PASS if this position has passed all filters, i.e., a call is made at this position. Otherwise,
        #       if the site has not passed all filters, a semicolon-separated list of codes for filters that fail. e.g. “q10;s50” might
        #       indicate that at this site the quality is below 10 and the number of samples with data is below 50% of the total
        #       number of samples. ‘0’ is reserved and should not be used as a filter String. If filters have not been applied,
        #       then this field should be set to the missing value. 
        #       - PASS: if more than 90% of samples have the genotype
        # vcf_header += f"##FILTER=<ID=q{int(include_metrics['QUAL']*100)},Description=\"Quality below {float(include_metrics['QUAL']*100)}\">\n"
        for s_val in filter_fields:
            vcf_header += f"##FILTER=<ID=s{s_val},Description=\"Less than {s_val}% of samples have data\">\n"

    if include_info:
        #   AC : allele count in genotypes, for each ALT allele, in the same order as listed
        #   AF : allele frequency for each ALT allele in the same order as listed: use this when estimated from primary data, not called genotypes
        #   AN : total number of alleles in called genotypes
        vcf_header += "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of Samples With Data\">\n" \
                      "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Allele count in genotypes, for each ALT allele, in the same order as listed\">\n" \
                      "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency for each ALT allele in the same order as listed\">\n" \
                      "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of alleles in called genotypes\">\n" \

    vcf_header += "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n" \
                  "#"
    for field in header_fields:
        vcf_header += f"{field}\t"
    for sample in samples_list:
        vcf_header += f'{sample}\t'
    vcf_header = vcf_header.rstrip('\t')  # remove trailing tab
    vcf_header += "\n"

    if save_path is None or save_path == '':
        chroms = [f'{c}' for c in chrom if c != "ERROR"]
        chroms = '-'.join(chroms)
        if 'MINGPT' in imputation_method.upper():
            model_dir = 'minGPT/'
        elif 'GPT' in imputation_method.upper():
            model_dir = 'GPT/'
        elif 'RANDOM' in imputation_method.upper():
            model_dir = 'random/'
        else:
            model_dir = ''
        save_path = os.path.join('data/generated/vcfs',
                            f'{model_dir}{imputation_method}.chr{chroms}.{len(samples_list)}_samples.vcf')
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w+') as f:
        f.writelines(vcf_header)
        f.writelines(vcf_body)
    print(f"Generated VCF saved to '{save_path}'.")
    if len(bad_muts) > 0:
        warnings.warn(f"Skipped {len(bad_muts)} bad mutations.")
    return save_path

############################# RANDOM GENERATION #############################
def generate_random_deletion(main_sequence, num_of_bp_to_delete):
    """
    """
    idx = []
    for i in range(num_of_bp_to_delete):
        max_len = len(main_sequence) - i
        idx.append(random.randrange(0, max_len))
    # print(f"Indices to delete: {idx}")
    for i in idx:
        main_sequence = main_sequence[:i] + main_sequence[i + 1:]
    return main_sequence

def generate_random_insertion(main_sequence, num_of_bp_to_add):
    """
    """
    idx = []
    nucleobases = ['A', 'C', 'G', 'T']
    for i in range(num_of_bp_to_add):
        max_len = len(main_sequence) + i
        idx.append(random.randrange(0, max_len))
    # print(f"Indices to add: {idx}")
    for i in idx:
        main_sequence = main_sequence[:i] + random.choice(nucleobases) + main_sequence[i:]
    return main_sequence

def generate_random_substitution(main_sequence, num_of_bp_to_change):
    """
    """
    nucleobases = ['A', 'C', 'G', 'T']
    idx = [random.randrange(0, len(main_sequence)) for _ in range(num_of_bp_to_change)]
    # print(f"Indices to change: {idx}")
    for i in idx:
        main_sequence = main_sequence[:i] + random.choice(nucleobases) + main_sequence[i+1:]
    return main_sequence

def get_decreasing_probabilities(max_range):
        """
        """
        probabilities = [1/(idx+1) for idx in range(max_range-1)]
        prob_weights = sum(probabilities)
        probabilities = [w/(prob_weights**i+1) for i, w in enumerate(probabilities)]
        s = sum(probabilities)
        added = False
        for i in range(len(probabilities)):
            if probabilities[i] < 1-s:
                probabilities.insert(i, 1-s)
                added = True
                break
        if not added:
            probabilities.append(1-s)
        return probabilities

def generate_random_vcf(num_samples=100, num_mutations=100, max_nucleobase_seq_length=29,
                        mutation_types=['deletion', 'insertion', 'substitution'], save_path=None,
                        chr=DEFAULT_CHR, save=False, to_VCF=False, verbose=True):

    """
    Generate a random corpus of mutations and samples in the form:

    {   sample_1: "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_0|1",    
        sample_2: "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_1|1",    
        sample_3: "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_0|1"   }
    
    Args:
        num_samples: The number of samples to generate.
        num_mutations: The number of mutations per sample.
        max_nucleobase_seq_length: The maximum length of the nucleobase sequence.
                                    Defaults to 29 based on the longest mutation present
                                    in the source 1000 Genomes Project data 
                                    (chr22:36868921:GTCCGGGGCCTTCAGAACCAGAAAGTCAA>G).
        mutation_types: The types of mutations to generate (deletion, insertion, substitution).
        save_path: The path to save the generated corpus.
        chr: The chromosome number to generate mutations for (defaults to DEFAULT_CHR).
        to_VCF: Whether to convert the generated corpus to VCF format.
        save: Whether to save the generated VCF file.

    Returns:
        dict: generated sample-wise mutations dictionary.
    """
    chr_length = CHR_LENGTHS[chr] # only single chromosome supported

    genotypes = {f'synth_{i+1}': '' for i in range(num_samples)}
    nucleobases = ['A', 'C', 'G', 'T']

    multiallelic = {'refs': [0, 0, 0],
                    'alts': [0, 0, 0]}

    if verbose:
        gen_log = tqdm.tqdm(range(num_mutations), 
                       desc=f"Generating {num_mutations} random mutations for {num_samples} samples")
    else:
        gen_log = range(num_mutations)
    for _ in gen_log:
        pos = random.randint(1, chr_length)

        # up to 3-allelic SNPs (NB: only either reference OR alternate alleles can be multiallelic)
        prob_multiallelic = [0.8, 0.15, 0.05]
        num_refs = random.choices(range(1, 4), weights=prob_multiallelic, k=1)[0]
        num_alts = random.choices(range(1, 4), weights=prob_multiallelic, k=1)[0] if num_refs == 1 else 1

        multiallelic['refs'][num_refs-1] += 1
        multiallelic['alts'][num_alts-1] += 1
        
        gt_alleles = [f'{i}' for i in range(num_alts+1)]
        
        # decaying probability of getting longer reference alleles
        prob_of_ref_length = get_decreasing_probabilities(max_nucleobase_seq_length)
        ref_len = random.choices(range(1, max_nucleobase_seq_length+1), 
                                weights=prob_of_ref_length, k=1)[0]
        original_ref = ''.join(random.choices(nucleobases, k=ref_len))
        refs = {original_ref}
        alts = set()

        # print(f"Generating {num_refs} from ref: {refs}")
        # generate random but plausible (multiallelic) reference alleles
        while len(refs) != num_refs:  # ensure new reference allele is added
            multi_ref_type = random.choice(mutation_types)

            if multi_ref_type == 'deletion' and len(original_ref) > 1:
                bp_to_del = random.choices(range(1, len(original_ref)), 
                                        weights=get_decreasing_probabilities(len(original_ref)-1),
                                        k=1)[0]
                refs.add(generate_random_deletion(original_ref, bp_to_del))
            elif multi_ref_type == 'insertion' and len(original_ref) < max_nucleobase_seq_length:
                bp_to_add = random.choices(range(1, max_nucleobase_seq_length-len(original_ref)+1), 
                                        weights=get_decreasing_probabilities(max_nucleobase_seq_length-len(original_ref)),
                                        k=1)[0]
                refs.add(generate_random_insertion(original_ref, bp_to_add))
            else:
                bp_to_change = random.choices(range(1, len(original_ref)+1), 
                                        weights=get_decreasing_probabilities(len(original_ref)),
                                        k=1)[0]
                refs.add(generate_random_substitution(original_ref, bp_to_change))
        
         # generate random but plausible (multiallelic) alternate alleles
        while len(alts) != num_alts:
            ref_to_mutate = random.choice(list(refs))
            mutation_type = random.choice(mutation_types)

            if mutation_type == 'deletion' and len(ref_to_mutate) > 1:
                bp_to_del = random.choices(range(1, len(ref_to_mutate)),
                                        weights=get_decreasing_probabilities(len(ref_to_mutate)-1),
                                        k=1)[0]
                alts.add(generate_random_deletion(ref_to_mutate, bp_to_del))
            elif mutation_type == 'insertion' and len(ref_to_mutate) < max_nucleobase_seq_length:
                bp_to_add = random.choices(range(1, max_nucleobase_seq_length-len(ref_to_mutate)+1),
                                        weights=get_decreasing_probabilities(max_nucleobase_seq_length-len(ref_to_mutate)),
                                        k=1)[0]
                alts.add(generate_random_insertion(ref_to_mutate, bp_to_add))
            else:
                bp_to_change = random.choices(range(1, len(ref_to_mutate)+1),
                                        weights=get_decreasing_probabilities(len(ref_to_mutate)),
                                        k=1)[0]
                alts.add(generate_random_substitution(ref_to_mutate, bp_to_change))
            
            if any(alt in refs for alt in alts):
                # print(f"WARNING: Found alternate allele(s) {alts} that are the same as reference allele(s) {refs}.")
                alts = {alt for alt in alts if alt not in refs}

        for sample in genotypes.keys():
            genotypes[sample] += ' '

            allele_1 = random.choice(gt_alleles)
            allele_2 = random.choice(gt_alleles)
            
            # genotypes[sample] += f"{chr}:{pos}_{ref}>{alt}_{allele_1}|{allele_2}"
            genotypes[sample] += f"{chr}:{pos}:{'/'.join(refs)}>{'/'.join(alts)}_{allele_1}|{allele_2}"
    
    # print(f"Multiallelic reference alleles: {multiallelic['refs']}")
    # print(f"Multiallelic alternate alleles: {multiallelic['alts']}")
    
    if to_VCF:
        corpus_to_VCF(genotypes, imputation_method='RandomModel', save_path=save_path)
    if save:
        if save_path is None or save_path == '':
            save_path = os.path.join('data/generated/json/random',
                                 f'rand_corpus_chr{chr}_{num_samples}ids_{num_mutations}muts.json')        
        with open(save_path, 'x') as f:
                json.dump(genotypes, f, sort_keys=False, indent=4)
        print(f"Generated corpus saved to '{save_path}'.")

    return genotypes

def generate_random_sample(num_mutations=100, max_nucleobase_seq_length=60, chr=DEFAULT_CHR,
                        mutation_types=['deletion', 'insertion', 'substitution']):

    """
    Generate a random sequence of mutations for a sample in the form: 
    
                "1:545363_T>C_0|0 1:793571_C>T_0|1 11:126186930_A>G_0|1 ..."
    
    Args:
        num_mutations: The number of mutations per sample.
        max_nucleobase_seq_length: The maximum length of the nucleobase sequence.
                                    Defaults to 29 based on the longest mutation present
                                    in the source 1000 Genomes Project data 
                                    (chr22:36868921:GTCCGGGGCCTTCAGAACCAGAAAGTCAA>G).
        mutation_types: The types of mutations to generate (deletion, insertion, substitution).
        chr: The chromosome number to generate mutations for (defaults to DEFAULT_CHR).

    Returns:
        str: generated sample-level mutations sequence.
    """
    chr_length = CHR_LENGTHS[chr] # only single chromosome supported
    nucleobases = ['A', 'C', 'G', 'T']
    genotype = ''
    multiallelic = {'refs': [0, 0, 0],
                    'alts': [0, 0, 0]}

    for _ in tqdm.tqdm(range(num_mutations), 
                       desc=f"Generating {num_mutations} random mutations"):
        pos = random.randint(1, chr_length)

        # up to 3-allelic SNPs (NB: only either reference or alternate alleles can be multiallelic)
        prob_multiallelic = [0.7, 0.25, 0.05]
        num_refs = random.choices(range(1, 4), weights=prob_multiallelic, k=1)[0]
        num_alts = random.choices(range(1, 4), weights=prob_multiallelic, k=1)[0] if num_refs == 1 else 1

        multiallelic['refs'][num_refs-1] += 1
        multiallelic['alts'][num_alts-1] += 1
        
        gt_alleles = [f'{i}' for i in range(num_alts+1)]
        
        # decaying probability of getting longer reference alleles
        prob_of_ref_length = get_decreasing_probabilities(max_nucleobase_seq_length)
        ref_len = random.choices(range(1, max_nucleobase_seq_length+1), 
                                weights=prob_of_ref_length, k=1)[0]
        original_ref = ''.join(random.choices(nucleobases, k=ref_len))
        refs = {original_ref}
        alts = set()

        # print(f"Generating {num_refs} from ref: {refs}")
        # generate random but plausible (multiallelic) reference alleles
        while len(refs) != num_refs:  # ensure new reference allele is added
            multi_ref_type = random.choice(mutation_types)

            if multi_ref_type == 'deletion' and len(original_ref) > 1:
                bp_to_del = random.choices(range(1, len(original_ref)), 
                                        weights=get_decreasing_probabilities(len(original_ref)-1),
                                        k=1)[0]
                refs.add(generate_random_deletion(original_ref, bp_to_del))
            elif multi_ref_type == 'insertion' and len(original_ref) < max_nucleobase_seq_length:
                bp_to_add = random.choices(range(1, max_nucleobase_seq_length-len(original_ref)+1), 
                                        weights=get_decreasing_probabilities(max_nucleobase_seq_length-len(original_ref)),
                                        k=1)[0]
                refs.add(generate_random_insertion(original_ref, bp_to_add))
            else:
                bp_to_change = random.choices(range(1, len(original_ref)+1), 
                                        weights=get_decreasing_probabilities(len(original_ref)),
                                        k=1)[0]
                refs.add(generate_random_substitution(original_ref, bp_to_change))
        
         # generate random but plausible (multiallelic) alternate alleles
        while len(alts) != num_alts:
            ref_to_mutate = random.choice(list(refs))
            mutation_type = random.choice(mutation_types)

            if mutation_type == 'deletion' and len(ref_to_mutate) > 1:
                bp_to_del = random.choices(range(1, len(ref_to_mutate)),
                                        weights=get_decreasing_probabilities(len(ref_to_mutate)-1),
                                        k=1)[0]
                alts.add(generate_random_deletion(ref_to_mutate, bp_to_del))
            elif mutation_type == 'insertion' and len(ref_to_mutate) < max_nucleobase_seq_length:
                bp_to_add = random.choices(range(1, max_nucleobase_seq_length-len(ref_to_mutate)+1),
                                        weights=get_decreasing_probabilities(max_nucleobase_seq_length-len(ref_to_mutate)),
                                        k=1)[0]
                alts.add(generate_random_insertion(ref_to_mutate, bp_to_add))
            else:
                bp_to_change = random.choices(range(1, len(ref_to_mutate)+1),
                                        weights=get_decreasing_probabilities(len(ref_to_mutate)),
                                        k=1)[0]
                alts.add(generate_random_substitution(ref_to_mutate, bp_to_change))
            
            if any(alt in refs for alt in alts):
                # print(f"WARNING: Found alternate allele(s) {alts} that are the same as reference allele(s) {refs}.")
                alts = {alt for alt in alts if alt not in refs}

        allele_1 = random.choice(gt_alleles)
        allele_2 = random.choice(gt_alleles)
        genotype += f"{chr}:{pos}:{'/'.join(refs)}>{'/'.join(alts)}_{allele_1}|{allele_2} "
    
    # print(f"Multiallelic reference alleles: {multiallelic['refs']}")
    # print(f"Multiallelic alternate alleles: {multiallelic['alts']}")
    
    return genotype