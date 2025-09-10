from metrics import *
from data.data_utils import *
import warnings
from tqdm import tqdm
from cyvcf2 import VCF

################################# Utility Evaluation #################################
UTILITY_FIGURES = 'figures/generated_data/utility_eval'
class UtilityEval:
    """
    """
    def __init__(self, model, ref_genomes, sample_wise=True, vcf_file=None, **kwargs):
        self.model = model
        self.model_name = kwargs.get('model_name', model.__class__.__name__)
        self.sample_wise = sample_wise
        self.vcf_file = vcf_file
        
        self.original_sequences = get_sequences_from_file(ref_genomes)
        self.original_sequences = [gt['genotypes'] for gt in self.original_sequences]
        self.original_sequences, _, _ = get_valid_sequences(self.original_sequences,
                                                            has_sample_names=True)

        self.AVAILABLE_METRICS = ['mut_validity', 'mut_quality', 'mut_memorization',
                                'mut_novelty', 'mut_uniqueness', 'mut_repetition',
                                'mut_distribution', 'mut_statistics', 'vcf_quality',]
        self.model_utility = {metric: None for metric in self.AVAILABLE_METRICS}
        self.METRIC_DESCRIPTION_DICT = {
            'mut_validity': 'Ratio of valid mutations out of all generated ones',
            'mut_quality': 'Ratio of locally valid mutations out of all (valid) generated ones',
            'mut_memorization': 'Number of memorized mutations (seen in training)',
            'mut_novelty': 'Number of novel mutations (unseen in training)',
            'mut_uniqueness': 'Number of unique mutations (amongst all generated samples)',
            'mut_repetition': 'Number of repeated mutations (common to more than one generated sample)',
            'mut_distribution': 'Positional distribution of generated mutations',
            'mut_statistics': 'Distribution of generated mutation types',
            'vcf_quality': 'Statistics of generated VCF file'}
        self.METRICS_EVAL_DICT = {
            'mut_validity': self.get_mutations_validity, 
            'mut_quality': self.get_locally_valid_mutations, 
            'mut_memorization': self.get_new_generated_mutations, 
            'mut_novelty': self.get_new_generated_mutations,
            'mut_uniqueness': self.get_mutation_uniqueness, 
            'mut_repetition': self.get_mutation_uniqueness, 
            'mut_distribution': self.get_mutations_distribution,
            'mut_statistics': self.get_mutations_statistics,
            'vcf_quality': self.evaluate_vcf_quality}
        
    def __str__(self):
        desc = f"Utility Evaluation for {self.model_name}:\n"
        desc += self.print_utility_statistics(self.AVAILABLE_METRICS, print_desc=True)
        return desc

    def print_utility_statistics(self, metrics, print_desc=False, **kwargs):
        desc = ''
        for metric in metrics:
            val = self.model_utility[metric]
            if val is not None:
                if metric == 'mut_distribution':
                    desc += f"{metric} (see image above):\n"
                    plot_variant_distribution(val, title = f"{self.model_name} Variant Distribution", 
                        num_bins = 50, plot_alts_only=False, colors=kwargs.get('color', 'cornflowerblue'),
                        save_path=self.get_save_filepath(metric) if kwargs.get('save', True) else None)
                elif isinstance(val, dict):
                    desc += f"{metric}:\n"
                    for sub_metric, sub_val in val.items():
                        desc += f"- {sub_metric}: {sub_val}\n"
                else:
                    desc += f"{metric}: {val}\n"
                if print_desc:
                    desc += f"----------> {self.METRIC_DESCRIPTION_DICT[metric]}\n"
                desc += f"\n"
        return desc

    def get_save_filepath(self, metric_name, filetype='pdf'):
        return f"{UTILITY_FIGURES}/{self.model_name}_{metric_name}.{filetype}"

    def evaluate_model_utility(self, samples_genotypes, metrics='all', **kwargs):
        """
        """
        verbose = kwargs.get('verbose', True)

        if metrics == 'all':
            metrics = self.AVAILABLE_METRICS
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif metrics in [[], '', None]:
            warnings.warn("No metrics provided for evaluation. Defaulting to...")
            metrics = self.AVAILABLE_METRICS

        if isinstance(samples_genotypes, str):
            assert samples_genotypes.endswith('.json'),\
                "Invalid file format. Please provide a .json file."
            # if sequences stored in .json file
            samples_genotypes = get_sequences_from_file(samples_genotypes)

        # ensure mut_quality is evaluated first
        print("Getting valid mutations for analysis...", end='')
        valid_sample_gts = self.METRICS_EVAL_DICT['mut_validity'](samples_genotypes)
        print("done!")

        eval_log = tqdm.tqdm(metrics, desc=f"Evaluating {self.model_name} utility")
        
        for i,metric in enumerate(eval_log):
            if metric not in self.AVAILABLE_METRICS:
                warnings.warn(f"Metric {metric} not recognized. Skipping...")
                metrics.remove(metric)
            else:    
                if self.model_utility[metric] is None:
                    eval_log.set_postfix_str(f'{metric} evaluation...')
                    try:
                        if metric == 'vcf_quality':
                            vcf_file = self.vcf_file
                            if vcf_file is not None and os.path.exists(vcf_file):
                                    input_data = vcf_file
                            else:
                                eval_log.write("\tNo valid VCF file path provided. Creating VCF file from passed genotypes...")
                                if not self.sample_wise:
                                    eval_log.write("\tNote: sample_wise is set to False. Generating VCF for all samples together.")
                                    valid_gts_for_vcf = self.METRICS_EVAL_DICT['mut_validity'](samples_genotypes,
                                                                                            sample_wise=True,
                                                                                            set_val=False)
                                else:
                                    valid_gts_for_vcf = valid_sample_gts
                                # valid_gts_for_vcf = self.METRICS_EVAL_DICT['mut_quality'](valid_gts_for_vcf,
                                #                                                         sample_wise=True)
                                # valid_gts_for_vcf = {f'synth_{i+1}': ' '.join(gt) for i,gt in enumerate(valid_gts_for_vcf)}
                                input_data = corpus_to_VCF(valid_gts_for_vcf,
                                                        imputation_method=self.model_name,
                                                        save_path=vcf_file, verbose=False)
                        elif metric in ['mut_uniqueness', 'mut_repetition']:
                            input_data = self.METRICS_EVAL_DICT['mut_validity'](samples_genotypes,
                                                                                sample_wise=True,
                                                                                set_val=False)
                        else:
                            input_data = valid_sample_gts
                        self.METRICS_EVAL_DICT[metric](input_data)
                    except Exception as e:
                        eval_log.write(f"Error evaluating {metric}: {e}. Skipping...")
                        self.model_utility[metric] = None
                eval_log.set_postfix_str(f'{metric} evaluation...done!')

        if verbose:
            print(f"\nUtility Evaluation for {self.model_name}:")
            print(self.print_utility_statistics(metrics, print_desc=True))

    ############################## Metric Calculations ##############################

    def get_mutations_validity(self, samples_genotypes, **kwargs):
        """
        Get valid sequences from the generated samples.
        """
        set_val=kwargs.get('set_val', True)
        sample_wise = kwargs.get('sample_wise', self.sample_wise)
        valid_sequences, valid_sequences_ratio, _ = get_valid_sequences(samples_genotypes,
                                                                                    sample_wise=sample_wise)
        if set_val:
            if sample_wise:
                self.model_utility['mut_validity'] = valid_sequences_ratio
            else:
                self.model_utility['mut_validity'] = np.mean(valid_sequences_ratio)
        return valid_sequences

    def get_locally_valid_mutations(self, samples_genotype, **kwargs):
        """
        """
        sample_wise = kwargs.get('sample_wise', self.sample_wise)
        if sample_wise:
            locally_valid_mutations = [get_locally_valid_mutations(sample) for sample in samples_genotype]
        else:
            locally_valid_mutations = get_locally_valid_mutations(samples_genotype)
        num_locally_valid = [len(v)/len(t) for v,t in zip(locally_valid_mutations, samples_genotype)]
        if sample_wise:
            self.model_utility['mut_quality'] = num_locally_valid
        else:
            self.model_utility['mut_quality'] = np.mean(num_locally_valid)
        return locally_valid_mutations

    def get_new_generated_mutations(self, samples_genotypes):
        unique_muts, memorized_muts, all_muts = get_new_generated_mutations(samples_genotypes,
                                                                  original_sequences=self.original_sequences,
                                                                  sample_wise=self.sample_wise)
        if self.sample_wise:
            num_memorized = [len(memoized)/len(muts) for memoized, muts in zip(memorized_muts, samples_genotypes)]
            num_unique = [len(unique)/len(muts) for unique, muts in zip(unique_muts, samples_genotypes)]
            self.model_utility['mut_memorization'] = num_memorized
            self.model_utility['mut_novelty'] = num_unique
        else:
            num_memorized = len(memorized_muts)/all_muts
            num_unique = len(unique_muts)/all_muts
            self.model_utility['mut_memorization'] = np.mean(num_memorized)
            self.model_utility['mut_novelty'] = np.mean(num_unique)

    def get_mutation_uniqueness(self, samples_genotypes):
        """
        Get mutation uniqueness from the generated samples.
        """
        _, mutation_uniqueness_ratio = get_uniqueness_score(samples_genotypes)
        if self.sample_wise:
            self.model_utility['mut_uniqueness'] = mutation_uniqueness_ratio
            self.model_utility['mut_repetition'] = [1-u for u in mutation_uniqueness_ratio]
        else:
            self.model_utility['mut_uniqueness'] = np.mean(mutation_uniqueness_ratio)
            self.model_utility['mut_repetition'] = np.mean([1-u for u in mutation_uniqueness_ratio])

    def get_mutations_distribution(self, samples_genotypes, chr=22):
        mut_positions_dist = get_sequences_by_chromosome(samples_genotypes, chr,
                                                        sample_wise=self.sample_wise)
        if self.sample_wise:
            chrom_dist = [get_dist_by_chrom(sample_dist,
                                       plot_alts_only=False) for sample_dist in mut_positions_dist]
        else:
            chrom_dist = get_dist_by_chrom(mut_positions_dist,
                                       plot_alts_only=False)
        self.model_utility['mut_distribution'] = chrom_dist

    def get_mutations_statistics(self, samples_genotypes):
        """
        Get mutation statistics from the generated samples.
        """
        mut_statistics = get_mutations_by_type(samples_genotypes)
        if self.sample_wise:
            self.model_utility['mut_statistics'] = mut_statistics
        else:
            statistics = {}
            tot = sum([len(muts) for muts in mut_statistics.values()]) / 2
            for category, muts in mut_statistics.items():
                statistics[category] = len(muts)/tot
            self.model_utility['mut_statistics'] = statistics

    def evaluate_vcf_quality(self, vcf_file):
        
        def get_filter_stats(filter_values):
            keys = set(filter_values)
            tot = len(filter_values)
            # NB: a value of PASS or ‘.’ in the INFO field will
            #     be 'None' with cyvcf2 package > replace
            keys.discard(None)
            keys.add('PASS')
            stats = {k:0 for k in keys}
            for val in filter_values:
                if val is None:
                    val = 'PASS'
                stats[val] += 1
            stats = {k: v/tot for k,v in stats.items()}
            return stats
        
        def get_gt_stats(genotypes):
            total = sum(genotypes.values())
            stats = {k: v/total for k,v in genotypes.items()}
            return stats

        def_keys = ['NS', 'AC', 'AF', 'AN', 'FILTER', 'call_rate']
        gts = {'homozygous_ref':0,
                'homozygous_alt':0,
                'heterozygous_ref':0}
        try:
            vcf_file = VCF(vcf_file, gts012=True)

            self.model_utility['vcf_quality'] = {}
            self.model_utility['vcf_quality']['FILTER'] = []
            self.model_utility['vcf_quality']['call_rate'] = []
            alt_allele_freqs = []
            for variant in vcf_file:
                try:
                    alt_allele_freqs.append(variant.aaf)
                    # INFO field
                    for allele1, allele2, _ in variant.genotypes:
                        if int(allele1) == 0 and int(allele2) == 0:
                            gts['homozygous_ref'] += 1
                        elif int(allele1) == 0 or int(allele2) == 0:
                            gts['heterozygous_ref'] += 1
                        else:
                            gts['homozygous_alt'] += 1
                    
                    for var_field,field_value in variant.INFO:
                        if var_field in def_keys:
                            for val in field_value if isinstance(field_value, (list, tuple)) else [field_value]:
                                if var_field not in self.model_utility['vcf_quality']:
                                    self.model_utility['vcf_quality'][var_field] = []
                                self.model_utility['vcf_quality'][var_field].append(val)
                    
                    # genotypes
                    self.model_utility['vcf_quality']['genotypes'] = gts
                    # FILTER field
                    self.model_utility['vcf_quality']['FILTER'].append(variant.FILTER)
                    # call rate
                    self.model_utility['vcf_quality']['call_rate'].append(variant.call_rate)
                except Exception as e:
                    tqdm.tqdm.write(f"{e}. Skipping...")
                    continue
            
            try:
                for var_field, field_value in self.model_utility['vcf_quality'].items():
                    if var_field == 'FILTER':
                        self.model_utility['vcf_quality'][var_field] = get_filter_stats(field_value)
                    elif var_field == 'genotypes':
                        self.model_utility['vcf_quality'][var_field] = get_gt_stats(field_value)
                    else:
                        self.model_utility['vcf_quality'][var_field] = np.mean(np.array(field_value, dtype=float))
            except Exception as e:
                tqdm.tqdm.write(f"End error {e}. Skipping...")
            
            self.model_utility['vcf_quality']['aaf'] = np.mean(alt_allele_freqs)
            vcf_file.close()
        
        except Exception as e:
            tqdm.tqdm.write(f"Error loading VCF file: {e}")
            for var_field in def_keys:
                self.model_utility['vcf_quality'][var_field] = None

################################# Privacy Evaluation #################################
from attacks import *
import torch.nn.functional as F

PRIVACY_FIGURES = 'figures/generated_data/privacy_eval'
DISTANCE_METRICS = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
                    'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching',
                    'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
class PrivacyEval:
    """
    """
    def __init__(self, model, tokenizer, ref_genomes, holdout_genomes, device='cpu',
                plot=True, save_plot=True, default_metric='cosine', **kwargs):
        self.model = model
        self.model_name = model.__class__.__name__
        self.tokenizer = tokenizer
        self.device = device

        self.plot = plot
        self.save_plot = save_plot
        self.dist_metric = default_metric

        self.training_gts = self._format_ref_gts(ref_genomes)
        self.holdout_gts = self._format_ref_gts(holdout_genomes)
        generated_samples = kwargs.get('synthetic_gts', None)
        if generated_samples is not None:
            self.synthetic_gts = json.load(open(generated_samples, 'r'))

        self.max_sample_length = kwargs.get('max_sample_length', 100)
        self.samples_to_evaluate = kwargs.get('samples_to_evaluate', 100)

        self.rare_muts_file = 'data/sources/rare_snps.txt'
        self.common_muts_file = 'data/sources/common_snps.txt'

        # self.AVAILABLE_ATTACKS = ['nn_matching', 'prompt_based_il',
        #                         'prompt_length_based_il', 'membership_inference',
        #                           'rare_variant_leakage', 'reconstruction']
        self.AVAILABLE_ATTACKS = ['prompt_based_il', 'prompt_length_based_il',
                                  'membership_inference', 'rare_variant_leakage']
                                #   'reconstruction']
        self.model_privacy = {attack: None for attack in self.AVAILABLE_ATTACKS}

        self.ATTACK_DESCRIPTION_DICT = {
            'prompt_based_il': 'Asserts if the model leaks information about samples in the training dataset by prompting it with training and holdout mutations', 
            'prompt_length_based_il': 'Asserts if the model leaks information based on the length of the prompt', 
            'membership_inference': 'Infers whether a specific sample was part of the training dataset', 
            'rare_variant_leakage': 'Asserts if the model reproduces rare variant combinations present in samples in the training dataset', }
        self.ATTACK_EVAL_DICT = {
            'prompt_based_il': self.prompt_based_il_attack, 
            'prompt_length_based_il': self.prompt_length_il_attack, 
            'membership_inference': self.membership_inference, # TODO
            'rare_variant_leakage': self.rare_variant_leakage, # TODO
            } 
        
    def _format_ref_gts(self, ref_genomes):
        genotypes = get_sequences_from_file(ref_genomes)
        genotypes = [gt['genotypes'] for gt in genotypes]
        genotypes, _, _ = get_valid_sequences(genotypes,
                                                    has_sample_names=True,
                                                    sample_wise=True)
        return genotypes
    
    def _get_tokenized_gts(self, genotypes):
        print("tokenizing reference genotypes...")
        if isinstance(self.tokenizer, RegexTokenizer):
            tokenized_gts = [self.tokenizer.encode(' '.join(gts), 
                                                allowed_special='all') for gts in genotypes]
        else:
            tokenized_gts = [self.tokenizer.encode(' '.join(gts)) for gts in genotypes]
        print("Done")
        return tokenized_gts

    def __str__(self):
        desc = f"Privacy Evaluation for {self.model_name}\n"
        desc += self.print_privacy_statistics(self.AVAILABLE_ATTACKS, print_desc=True)
        return desc

    def print_privacy_statistics(self, attacks, print_desc=False):
        desc = ''
        for attack in attacks:
            val = self.model_privacy[attack]
            if val is not None:
                if isinstance(val, dict):
                    desc += f"{attack}:\n"
                    for sub_metric, sub_val in val.items():
                        if isinstance(sub_val, dict):
                            desc += f"- {sub_metric}:\n"
                            for sub_sub_metric, sub_sub_val in sub_val.items():
                                desc += f"\t- {sub_sub_metric}: {sub_sub_val}\n"
                        else:
                            desc += f"- {sub_metric}: {sub_val}\n"
                else:
                    desc += f"{attack}: {val}\n"
                if print_desc:
                    desc += f"----------> {self.ATTACK_DESCRIPTION_DICT[attack]}\n"
                desc += f"\n"
        return desc
    
    def get_save_filepath(self, attack_name, filetype='pdf'):
        return f"{PRIVACY_FIGURES}/{self.model_name}_{attack_name}.{filetype}"

    def evaluate_privacy(self, attacks='all', **kwargs):
        """
        Evaluate the privacy of the generated samples.
        """
        verbose = kwargs.get('verbose', True)
        if attacks == 'all':
            attacks = self.AVAILABLE_ATTACKS
        elif isinstance(attacks, str):
            attacks = [attacks]
        elif attacks in [[], '', None]:
            warnings.warn("No attack provided for evaluation. Defaulting to...")
            attacks = self.AVAILABLE_ATTACKS

        eval_log = tqdm.tqdm(attacks, desc=f"Evaluating {self.model_name} privacy")
        
        for i,attack in enumerate(eval_log):
            if attack not in self.AVAILABLE_ATTACKS:
                warnings.warn(f"Attack {attack} not supported. Skipping...")
                attacks.remove(attack)
            else:    
                if self.model_privacy[attack] is None:
                    eval_log.set_postfix_str(f'{attack} evaluation...')
                    try:
                        self.ATTACK_EVAL_DICT[attack](**kwargs)
                    except Exception as e:
                        tqdm.tqdm.write(f"Error evaluating {attack}: {e}. Skipping...")
                        self.model_privacy[attack] = None
                eval_log.set_postfix_str(f'{attack} evaluation...done!')

        if verbose:
            print(f"\nPrivacy Evaluation for {self.model_name}:")
            print(self.print_privacy_statistics(attacks, print_desc=True))

    ############################## Metric Calculations ##############################
    
    def generate_prompts(self, prompt_length=1, **kwargs):
        available_prompt_types = ['random', 'training', 'holdout']

        prompt_types = kwargs.get('prompt_types', ['training', 'holdout'])
        samples_per_prompt = kwargs.get('samples_per_prompt', 1)

        if len(prompt_types) == 0:
            warnings.warn("No prompt types provided. Defaulting to 'training'.")
            prompt_types = ['training']
        if any([p not in available_prompt_types for p in prompt_types]):
            warnings.warn(f"Invalid prompt types provided. Defaulting to 'training'.")
            prompt_types = ['training']

        prompting_data = {k: {'prompts': [], 'samples': [] } for k in prompt_types}

        if 'random' in prompt_types:
            for _ in range(samples_per_prompt):
                random_sample = generate_random_vcf(num_samples=1, 
                                                    num_mutations=self.max_sample_length,
                                                    verbose=False)['synth_1']
                random_prompt = ' '.join(re.findall(SEQUENCE_PATTERN,
                                                    random_sample)[:prompt_length])
                prompting_data['random']['prompts'].append(random_prompt)
                prompting_data['random']['samples'].append(self.tokenizer.encode(random_sample))

        # if 'synthetic' in prompt_types:
        #     for _ in range(samples_per_prompt):
        #         synth_sample = query_model(prompt='22:', tokenizer=self.tokenizer, 
        #                                     model=self.model, 
        #                                     mutations_to_generate=self.max_sample_length, 
        #                                     return_tokens=True, custom=True
        #                                     ).detach().cpu().numpy()
        #         synth_prompt = self.tokenizer.encode(synth_sample)
        #         synth_prompt = ' '.join(re.findall(SEQUENCE_PATTERN, synth_sample)[:prompt_length])
        #         prompting_data['synthetic']['prompts'].append(synth_prompt)
        #         prompting_data['synthetic']['samples'].append(synth_sample)

        training_ids = kwargs.get('training_ids',
                                random.choices(range(len(self.training_gts)),
                                               k=samples_per_prompt))
        if 'training' in prompt_types:
            for i in training_ids:
                training_sample = self.training_gts[i]
                training_prompt = ' '.join(training_sample[:prompt_length])
                training_sample = self.tokenizer.encode(' '.join(training_sample))

                prompting_data['training']['prompts'].append(training_prompt)
                prompting_data['training']['samples'].append(training_sample)


        holdout_ids = kwargs.get('holdout_ids',
                                random.choices(range(len(self.holdout_gts)),
                                               k=samples_per_prompt))
        if 'holdout' in prompt_types:
            for i in holdout_ids:
                holdout_sample = self.holdout_gts[i]
                holdout_prompt = ' '.join(holdout_sample[:prompt_length])
                holdout_sample = self.tokenizer.encode(' '.join(holdout_sample))

                prompting_data['holdout']['prompts'].append(holdout_prompt)
                prompting_data['holdout']['samples'].append(holdout_sample)
        
        return prompting_data

    def prompt_based_il_attack(self, **kwargs):
        prompt_length = kwargs.get('prompt_length', 1)
        prompt_types = kwargs.get('prompt_types', ['training', 'holdout'])
        if not ('training' in prompt_types and 'holdout' in prompt_types):
            warnings.warn("Prompt types must include AT LEAST 'training' and 'holdout'. Using defaults...")
            prompt_types = ['training', 'holdout']

        prompting_data = self.generate_prompts(**kwargs)

        _, distances = prompt_based_information_leakage(
                        target_model=self.model,
                        tokenizer=self.tokenizer,
                        max_sample_length=self.max_sample_length,
                        prompting_data=prompting_data,
                        prompt_length=prompt_length,
                        distance_metric=self.dist_metric, 
                        **kwargs)
        self.model_privacy['prompt_based_il'] = distances # [samples, rounds]

    def prompt_length_il_attack(self, max_prompt_length, **kwargs):
        prompt_types = kwargs.get('prompt_types', ['training', 'holdout'])
        if 'training' not in prompt_types:
            warnings.warn("Prompt types must include AT LEAST 'training'. Using defaults...")
            prompt_types = ['training']

        self.model_privacy['prompt_length_based_il'] = {l: [] for l in range(1, max_prompt_length + 1)}

        for prompt_length in range(1, max_prompt_length+1):
            prompting_data = self.generate_prompts(prompt_length, **kwargs)

            _, distances = prompt_based_information_leakage(
                            target_model=self.model,
                            tokenizer=self.tokenizer,
                            max_sample_length=self.max_sample_length,
                            prompting_data=prompting_data,
                            prompt_length=prompt_length,
                            distance_metric=self.dist_metric, 
                            **kwargs)
            # shape [prompts, lengths, samples, rounds]
            self.model_privacy['prompt_length_based_il'][prompt_length] = distances
        
        # from {lenX: {prompt_type: [distX]}, ...} to {prompt_type: [dist1, dist2, ...]}
        final_distances = self.model_privacy['prompt_length_based_il']
        distances_by_len = {prompt_type: [v[prompt_type] 
                                        for v in list(final_distances.values())] 
                            for prompt_type in prompt_types}
        self.model_privacy['prompt_length_based_il'] = distances_by_len
    
    def plot_mia_results(self, results, save_path=None, cmap='plasma', **kwargs):
        results_to_plot= [[
            mia_results['auc'],
            mia_results['accuracy'],
            mia_results['precision'],
            mia_results['recall'],
            mia_results['fscore'],
            mia_results['attack_advantage']
        ] for mia_results in results.values()]
        
        title = kwargs.get('title', f"{self.model_name} MIA Results")

        plot_metrics_by_model(results_to_plot, model_names=['AUC', 'Accuracy', 'Precision', 
                                                        'Recall', 'F1 Score', 'Attack Advantage'], 
                                                labels = ['Threshold', 'Logistic\nRegression',
                                                            'Random Forest', 'KNN'],
                                                legend_title='MIA metric',
                                                x_ticks=['AUC', 'Accuracy', 'Precision', 
                                                        'Recall', 'F1 Score', 'Attack Advantage'],
                                                xlabel = 'Attack', ylabel='Score',
                                                title=title, 
                                                save_path=save_path, 
                                                cmap=cmap,
                                                bbox_to_anchor=kwargs.get('bbox_to_anchor', (0.8, 0.9)))

    def membership_inference(self, train_sequences, test_sequences, **kwargs):
        """
            AUC > 0.7: High privacy risk - model is memorizing training data
            AUC 0.5-0.7: Moderate risk - some overfitting present
            AUC ≈ 0.5: Low risk - good privacy preservation
        """
        from attacks import run_all_mia_attacks, analyze_feature_importance, privacy_risk_assessment, plot_mia_results, plot_mia_comparison
        """
        Complete manual MIA evaluation pipeline
        
        Args:
            model: Trained minGPT model
            tokenizer: Model tokenizer  
            train_sequences: Training sequences (members)
            test_sequences: Test sequences (non-members)
            
        Returns:
            Tuple of (results_dict, privacy_assessment_string)
        """
        results, member_features, non_member_features = run_all_mia_attacks(self.model, 
                                                                            self.tokenizer, 
                                                                            train_sequences, 
                                                                            test_sequences, 
                                                                            **kwargs)
        # Plot results
        if kwargs.get('plot', self.plot):
            # Generate privacy assessment
            assessment = privacy_risk_assessment(results)
            print(assessment)

            # Feature importance analysis
            # feature_importance = analyze_feature_importance(member_features, non_member_features)
            
            # print("\nFeature Importance for MIA:")
            # print("="*40)
            # for feature, importance in sorted(feature_importance.items(), 
            #                                 key=lambda x: x[1], reverse=True):
            #     print(f"{feature:20} | {importance:.4f}")

            # Plot results
            plot_mia_results(results, **kwargs)
        
        return results

    def rare_variant_leakage(self, num_samples=20, **kwargs):
        """
        Rare variants have alt AF < 0.05, this means that reproducing homo ref genotype
        does not represent a privacy risk, but homo alt/het does.
        """
        samples_genotypes = kwargs.get('synthetic_gts', self.synthetic_gts)
        if isinstance(samples_genotypes, str) and samples_genotypes.endswith('.json'):
            with open(samples_genotypes, 'r') as f:
                samples_genotypes = json.load(f)
        elif isinstance(samples_genotypes, list):
            samples_genotypes = {f'synth_{i+1}': s for i, s in enumerate(samples_genotypes)}
        elif not isinstance(samples_genotypes, dict):
            raise ValueError("samples_genotypes must be a dict or a path to a .json file.")

        valid_gts, _, _ = get_valid_sequences(list(samples_genotypes.values())[:num_samples],
                                            sample_wise=True)

        rare_muts = open(self.rare_muts_file, 'r').readlines()
        rare_muts = [mut.strip('\n') for mut in rare_muts]
        common_muts = open(self.common_muts_file, 'r').readlines()
        common_muts = [mut.strip('\n') for mut in common_muts]

        memorized_rare = [0 for _ in valid_gts]
        memorized_common = [0 for _ in valid_gts]
        leaked_rare_muts = [0 for _ in valid_gts]
        leaked_common_muts = [0 for _ in valid_gts]
        non_memorized = [0 for _ in valid_gts]
        memorized = [0 for _ in valid_gts]

        analysis_log = tqdm.tqdm(valid_gts)
        for i, sample in enumerate(analysis_log):
            unk = 0
            analysis_log.set_description(f"Analyzing {len(sample)} mutations")
            for mut in sample:
                mut, gt = mut.split('_')
                if mut in rare_muts:
                    memorized_rare[i] += 1
                    if gt != '0|0':
                        # these are privacy concerns
                        leaked_rare_muts[i] += 1
                if mut in common_muts:
                    memorized_common[i] += 1
                    if gt != '0|0':
                        leaked_common_muts[i] += 1
                else: 
                    unk += 1

            desc = f"Found {memorized_rare[i]} rare mutations"
            desc += f" (of which {leaked_rare_muts[i]} leaked non-ref)"
            desc += f" and {memorized_common[i]} common mutations"
            desc += f" (of which {leaked_common_muts[i]} leaked non-ref)"
            desc += f" and {unk} unseen mutations"
            analysis_log.set_postfix_str(desc)


            memorized_rare[i] = memorized_rare[i]/len(sample)
            memorized_common[i] = memorized_common[i]/len(sample)
            leaked_rare_muts[i] = leaked_rare_muts[i]/len(sample)
            leaked_common_muts[i] = leaked_common_muts[i]/len(sample)
            non_memorized[i] = unk/len(sample)
            memorized[i] = (len(sample) - non_memorized[i]) / len(sample)

        self.model_privacy['rare_variant_leakage'] = {
                        'memorized': np.mean(memorized),
                        'non_memorized': np.mean(non_memorized),
                        'memorized_rare': np.mean(memorized_rare),
                        'leaked_rare': np.mean(leaked_rare_muts),
                        'memorized_common': np.mean(memorized_common),
                        'leaked_common': np.mean(leaked_common_muts)}