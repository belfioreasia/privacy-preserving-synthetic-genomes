import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import warnings
import torch

from models.tokenizers import RegexTokenizer
from models.finetuning import GPT_SPECIAL_TOKENS
from data.data_utils import SEQUENCE_PATTERN, SNP_PATTERN, GT_PATTERN, CHR_LENGTHS, DEFAULT_CHR, corpus_to_VCF

def query_model(prompt, tokenizer, model, mutations_to_generate=100, 
                save_vcf=False, **kwargs):
    """
    Query the model with a sequence and return the generated sequence.
    """
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    
    model.eval()
    tokens_to_generate = mutations_to_generate * 10
    # print("Generated sequence length:", generated_seq_length)
    return_tokens = kwargs.get('return_tokens', True)
    custom_model = kwargs.get('custom', True)
    device = kwargs.get('device', 'cpu')

    if isinstance(prompt, str):
        if isinstance(tokenizer, RegexTokenizer):
            tokenized_prompt = tokenizer.encode(prompt, allowed_special='all')
            tokenized_prompt = torch.LongTensor(tokenized_prompt).unsqueeze(0)
        else:
            tokenized_prompt = torch.LongTensor([tokenizer.encode(prompt)])
    else:
        tokenized_prompt = prompt
    tokenized_prompt = tokenized_prompt.to(device)
    model = model.to(device)

    if custom_model:
        generated_tokens = model.generate(tokenized_prompt, tokens_to_generate)[0]
        if return_tokens:
            return generated_tokens
        else:
            generated_seqs = tokenizer.decode(generated_tokens.tolist())
    else:
        from models.finetuning import generate_sample
        from data.dataset import GPTDataFormatter
        formatter = GPTDataFormatter(custom=True)
        generated_tokens = generate_sample(model, tokenizer, formatter, prompt=prompt, 
                                        max_sample_length=tokens_to_generate, 
                    samples_to_generate=1, skip_special_tokens=False, custom=True,
                    temperature=1.0, return_tensors=return_tokens)[0]
        if return_tokens:
            return generated_tokens
        else:
            generated_seqs = generated_tokens.replace(GPT_SPECIAL_TOKENS['mutation_sep'], ' ')
        
    if save_vcf:
        model_name = type(model).__name__
        corpus_to_VCF(generated_seqs, imputation_method=model_name)
    return generated_seqs

################################# PLOTTING ################################# 
def plot_metrics_by_model(metrics, title, labels, xlabel, ylabel, figsize=(10, 6), 
                          save_path=None, **kwargs):
    """
    """
    fig, axs = plt.subplots(figsize=figsize, layout='constrained')
    # plt.gca().set_prop_cycle(None)

    color_selection = np.arange(len(metrics))
    cmap = mpl.colormaps[kwargs.get('cmap', 'plasma')] # ['Pastel1']
    colors = cmap(np.linspace(0, 1, len(color_selection)))
    # palette = plt.get_cmap("Pastel1")
    # colors = palette(color_selection)
    
    x_values = np.arange(len(metrics[0]))
    bar_width = 1. / len(metrics) - 0.05 # deduct some space between bars
    shift_ratio = 0

    plot_type = kwargs.get('plot_type', 'bar')
    plot_error_bar = kwargs.get('plot_error_bar', False)
    plot_max = kwargs.get('plot_max', not plot_error_bar)

    share_axis = kwargs.get('share_axis', True)
    if not share_axis:
        twin_ax = [axs]
        twin_ax.extend([axs.twinx() for _ in range(len(metrics)-1)])
        axs = twin_ax
            
    if 'x_ticks' in kwargs:
        from matplotlib import axes, ticker
        x_ticks = kwargs['x_ticks']
        # axs.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axs.set_xticks([i for i in range(len(x_ticks))])
        axs.set_xticklabels(x_ticks)
    else:
        plt.xticks([])

    for i, model_metric in enumerate(metrics):
        if plot_type == 'bar':
            offset = bar_width * shift_ratio
            mean = np.mean(model_metric, axis=0)
            if plot_error_bar:
                yerr = [mean - np.min(model_metric, axis=0), np.max(model_metric, axis=0) - mean]
                model_metric = mean
            else:
                yerr = [np.zeros_like(model_metric), np.zeros_like(model_metric)]
                if not kwargs.get('plot_original', True):
                    if isinstance(model_metric, list):
                        if plot_max:
                            model_metric = np.max(model_metric, axis=0)
                        else:
                            model_metric = np.mean(model_metric, axis=0)
            if not share_axis:
                if plot_error_bar:
                    metric_bar = axs[i].bar(x_values + offset, model_metric, bar_width, 
                        label=labels[i], color=colors[i], yerr=yerr)
                else:
                    metric_bar = axs[i].bar(x_values + offset, model_metric, bar_width, 
                            label=labels[i], color=colors[i])
                axs[i].bar_label(metric_bar, padding=5)
                axs[i].tick_params(axis='y', labelcolor=colors[i])
            else:
                if plot_error_bar:
                    metric_bar = axs.bar(x_values + offset, model_metric, bar_width, 
                            label=labels[i], color=colors[i], yerr=yerr)
                else:
                    metric_bar = axs.bar(x_values + offset, model_metric, bar_width, 
                            label=labels[i], color=colors[i])
                # axs.bar_label('{:.2%}'.format(metric_bar.value), padding=3)
                # axs.bar_label(metric_bar, padding=5)
            shift_ratio += 1
        elif plot_type == 'line':
            if not share_axis:
                axs[i].plot(x_values, model_metric, color=colors[i], label=labels[i], linewidth=2.)
                axs[i].tick_params(axis='y', labelcolor=colors[i])
            else:
                axs.plot(x_values, model_metric, color=colors[i], label=labels[i], linewidth=2.)
    if kwargs.get('set_max_y', False):
        # set max y-axis limit
        axs.set_ylim(top=1.0)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(labels) > 1 and kwargs.get('show_legend', True): 
        legend_title = kwargs.get('legend_title', 'Generation Model')
        bbox_to_anchor = kwargs.get('bbox_to_anchor', (0.8, 0.6))
        if not share_axis:
            fig.legend(title=legend_title, title_fontproperties={'weight':'bold'},
                       bbox_to_anchor=bbox_to_anchor)
        else:
            axs.legend(title=legend_title, title_fontproperties={'weight':'bold'},
                       bbox_to_anchor=bbox_to_anchor)

    if save_path:
        plt.savefig(f'{save_path}.pdf', format='pdf')
    plt.show()

def get_dist_by_chrom(muts_list, plot_alts_only, has_gts=True):
        """
        Get the distribution of mutations by chromosome.
        """
        if plot_alts_only and not has_gts:
            warnings.warn("plot_alts_only=True requires has_gts=True. Setting plot_alts_only = False")
            plot_alts_only = False

        present_chrs = {}
        for seq in muts_list:
            mut_chr, mut_pos, alleles = seq.split(':')
            mut_chr = int(mut_chr)
            mut_pos = int(mut_pos)
            if mut_chr not in list(present_chrs.keys()):
                if plot_alts_only:
                    alleles, gt = alleles.split('_')
                    if gt not in ['0/0', '0|0']:
                        # print(f"Adding chromosome {mut_chr} with position {mut_pos}")
                        present_chrs[mut_chr] = [mut_pos]
                else:
                    # print(f"Adding chromosome {mut_chr} with position {mut_pos}")
                    present_chrs[mut_chr] = [mut_pos]
            else:
                if plot_alts_only:
                    alleles, gt = alleles.split('_')
                    if gt not in ['0/0', '0|0']:
                        # print(f"Adding chromosome {mut_chr} with position {mut_pos}")
                        present_chrs[mut_chr].append(mut_pos)
                else:
                    present_chrs[mut_chr].append(mut_pos)
        return present_chrs

def plot_variant_distribution(muts_list, title, by_chr=True, plot_alts_only=False,
                            num_bins=100, sharey=True, save_path=None, **kwargs):
    """
    Plot the distribution of variants by model.
    """
    muts_by_chrom = {}
    palette = plt.get_cmap("Pastel1")
    has_gts = kwargs.get('has_gts', True)

    if isinstance(muts_list, dict):
        muts_by_chrom = {int(chr):muts for chr,muts in muts_list.items()}
        share_axis = True
        compare_dists = False
        alpha = 1.0
        color_selection = np.arange(1)
    elif isinstance(muts_list[0], list):
        assert 'model_names' in kwargs and 'share_axis' in kwargs, \
            "If multiple models are provided, 'model_names' and 'share_axis' must be specified."

        model_names = kwargs['model_names']
        share_axis = kwargs['share_axis']

        compare_dists = True
        alpha = 0.7
        for i, model_muts in enumerate(muts_list):
            if isinstance(plot_alts_only, list):
                if len(plot_alts_only) != len(muts_list):
                    plot_alts_only = [plot_alts_only[0]] * len(muts_list)
                # plot each metric differently
                plot_alts = plot_alts_only[i]
            else:
                plot_alts = plot_alts_only
            # dict of dicts
            if isinstance(model_muts, list) and isinstance(model_muts[0], (str, int)):
                muts_by_chrom[model_names[i]] = model_muts
            else:
                muts_by_chrom[model_names[i]] = get_dist_by_chrom(model_muts,
                                                            plot_alts,
                                                            has_gts=has_gts)
        color_selection = np.arange(len(model_names))
    else:
        if isinstance(plot_alts_only, list):
            if len(plot_alts_only) != len(muts_list):
                plot_alts_only = [plot_alts_only[0]] * len(muts_list)
            # plot each metric differently
            plot_alts = plot_alts_only[i]
        else:
            plot_alts = plot_alts_only

        share_axis = True
        compare_dists = False
        alpha = 1.0
        muts_by_chrom = get_dist_by_chrom(muts_list, plot_alts,
                                        has_gts=has_gts)
        color_selection = np.arange(1)
    colors = kwargs.get('colors', palette(color_selection))

    if by_chr:
        if compare_dists:
            unique_chroms = set()
            for model_name in model_names:
                unique_chroms.update(muts_by_chrom[model_name].keys())
        else:
            unique_chroms = set(muts_by_chrom.keys())
        num_present_chr = len(unique_chroms)
        # print(f"{num_present_chr} present chromosomes: {unique_chroms}")

        nrows = max(1, num_present_chr // 2)
        ncols = (num_present_chr + nrows - 1) // nrows
        num_x_ticks = min(20, num_bins)

        fig, axs = plt.subplots(figsize = (15, 6*nrows), nrows=nrows, ncols=ncols, 
                                sharey=sharey, layout='constrained')

        i = j = 0
        plt.suptitle(title,size=20)
        non_plotted = []
        for mut_chr in unique_chroms:
            if compare_dists:
                curr_chrom_muts = []
                # Get the mutations for the current chromosome from each model
                curr_chrom_muts = [muts_by_chrom[model_name][mut_chr] \
                                    if mut_chr in muts_by_chrom[model_name].keys() \
                                    else [0] \
                                    for model_name in model_names]
                non_plotted = [[mut for mut in model_chrom if mut > CHR_LENGTHS[int(mut_chr)]] \
                               for model_chrom in curr_chrom_muts]
                curr_chrom_muts = [[mut for mut in model_chrom if mut <= CHR_LENGTHS[int(mut_chr)]] \
                               for model_chrom in curr_chrom_muts]
                mins = [min(muts) for muts in curr_chrom_muts]
                min_pos = max(min(mins), 0)
                maxes = [max(muts) for muts in curr_chrom_muts]
                max_pos = min(max(maxes), CHR_LENGTHS[int(mut_chr)])
                # max_pos = max(curr_chrom_muts)
            else:
                curr_chrom_muts = muts_by_chrom[mut_chr]
                non_plotted = [mut for mut in curr_chrom_muts if mut > CHR_LENGTHS[int(mut_chr)]]
                curr_chrom_muts = [mut for mut in curr_chrom_muts if mut <= CHR_LENGTHS[int(mut_chr)]]
                min_pos = max(min(curr_chrom_muts), 0) # 0
                max_pos = min(max(curr_chrom_muts), CHR_LENGTHS[int(mut_chr)]) # CHR_LENGTHS[mut_chr]
                # max_pos = max(curr_chrom_muts)
            # print(len(non_plotted), "mutations exceed max position")
            step = max_pos // num_x_ticks
            if (step > max_pos - min_pos) and (max_pos - min_pos > 0):
                step = max_pos - min_pos

            x_ticks = np.arange(min_pos, max_pos, step) if step > 0 else np.array([min_pos, max_pos], dtype=float)
            # print(f'Chromosome {mut_chr} has {len(curr_chrom_muts)} mutations')
            # print(f'Row {i // ncols}, Column {j} of {nrows} rows and {ncols} columns')

            if nrows == 1 and ncols == 1:
                if not share_axis and compare_dists:
                    twin_ax = [axs]
                    twin_ax.extend([axs.twinx() for _ in range(len(model_names)-1)])
                    for k, ax in enumerate(twin_ax):
                        ax.hist(np.array(curr_chrom_muts[k], dtype=float),
                                bins=num_bins, alpha=alpha, 
                                label=model_names[k], color=colors[k])
                        ax.tick_params(axis='y', labelcolor=colors[k])

                else:
                    axs.hist(np.array(curr_chrom_muts, dtype=float), bins=num_bins,
                            alpha=alpha, color=colors)
                    print(f'Plotted {len(curr_chrom_muts)} mutations for chromosome {mut_chr}')
                axs.set_xticks(x_ticks)
                axs.set_xticklabels(x_ticks, rotation=45)

                axs.set_title(f'Chromosome {mut_chr}')
                axs.set_xlabel('Position')
                axs.set_ylabel('Number of Variants')

            elif nrows == 1:
                if not share_axis and compare_dists:
                    twin_ax = [axs[j]]
                    twin_ax.extend([axs[j].twinx() for _ in range(len(model_names)-1)])
                    for k, ax in enumerate(twin_ax):
                        ax.hist(np.array(curr_chrom_muts[k], dtype=float),
                                bins=num_bins, alpha=alpha, 
                                label=model_names[k], color=colors[k])
                        ax.tick_params(axis='y', labelcolor=colors[k])
                else:
                    axs[j].hist(curr_chrom_muts, bins=num_bins,
                                alpha=alpha, color=colors)

                axs[j].set_xticks(x_ticks)
                axs[j].set_xticklabels(x_ticks, rotation=45)

                axs[j].set_title(f'Chromosome {mut_chr}')
                axs[j].set_xlabel('Position')
                axs[j].set_ylabel('Number of Variants')

            else:
                if not share_axis and compare_dists:
                    twin_ax = [axs[i][j]]
                    twin_ax.extend([axs[i][j].twinx() for _ in range(len(model_names)-1)])
                    for k, ax in enumerate(twin_ax):
                        ax.hist(np.array(curr_chrom_muts[k], dtype=float),
                                bins=num_bins, alpha=alpha, 
                                label=model_names[k], color=colors[k])
                        ax.tick_params(axis='y', labelcolor=colors[k])
                else:
                    axs[i][j].hist(curr_chrom_muts, bins=num_bins,
                                   alpha=alpha, color=colors)
                axs[i][j].set_xticks(x_ticks)
                axs[i][j].set_xticklabels(x_ticks, rotation=45)

                axs[i][j].set_title(f'Chromosome {mut_chr}')
                axs[i][j].set_xlabel('Position')    
                axs[i][j].set_ylabel('Number of Variants')
                
            if compare_dists:
                fig.legend(title=kwargs.get('legend_title', 'Generation Model'),
                           title_fontproperties={'weight':'bold'})

            j += 1
            if j >= ncols:
                j = 0
                i += 1
    else:
        for chr in range(1, 23):
            if compare_dists:
                all_mutations = {}
                for model_name in model_names:
                    all_mutations[model_name] = [len(muts_by_chrom[model_name][chr])
                                    if chr in muts_by_chrom[model_name].keys() \
                                    else 0 \
                                    for chr in range(1, 23)]
            else:
                all_mutations = [0 for _ in range(22)]
                all_mutations[chr-1] = len(muts_by_chrom[chr]) if chr in muts_by_chrom.keys() else 0
            # print(f"Chromosome {chr} has {all_mutations[chr-1]} mutations")

        fig, axs = plt.subplots(figsize = (15, 6), layout='constrained')
        xticks = [chr for chr in range(1, 23)]
        if compare_dists:
            if not share_axis:
                twin_ax = [axs]
                twin_ax.extend([axs.twinx() for _ in range(len(model_names)-1)])
                for k, ax in enumerate(twin_ax):
                    ax.bar(xticks, all_mutations[model_names[k]], alpha=alpha, 
                            label=model_names[k], color=colors[k])
                    ax.tick_params(axis='y', labelcolor=colors[k])
            else:
                for k, model_name in enumerate(model_names):
                    axs.bar(xticks, all_mutations[model_name], alpha=alpha, 
                            label=model_name, color=colors[k])
            fig.legend(title=kwargs.get('legend_title', 'Generation Model'),
                       title_fontproperties={'weight':'bold'})
        else:
            axs.bar(xticks, all_mutations)
        
        axs.set_xticks([i for i in range(1, 23)])
        axs.set_xticklabels(xticks)

        axs.set_title('All Chromosomal variants')
        axs.set_xlabel('Chromosome')
        axs.set_ylabel('Number of Variants')

    if save_path:
        plt.savefig(f'{save_path}.pdf', format='pdf')
    plt.show()
    
def plot_sample_similarity(original_samples, model, tokenizer, metric='cosine', prompt_len=1, 
                           sample_len='all', num_samples=10, figsize=(15,15), save_path=None):
    """
    Plot the similarity between original and generated samples.
    """
    fig, ax = plt.subplots(figsize=figsize)

    original_feats = []
    generated_feats = []
    if sample_len == 'all':
        sample_len = min([len(s) for s in original_samples]) + 1
    else:
        assert isinstance(sample_len, int), "sample_len should be 'all' or an integer."
    
    if isinstance(num_samples, int):
        sample_idx = random.sample(range(len(original_samples)), num_samples)
    elif isinstance(num_samples, list):
        sample_idx = num_samples
    # print(f'Selected sample indices: {sample_idx}')

    for i in sample_idx:
        sample = original_samples[i]
        original_feats.append(tokenizer.encode(' '.join(sample))[:sample_len])
        # original_feats.append(extract_sample_features(sample))
        
        prompt = ' '.join(sample[:prompt_len])
        synt_gen = query_model(prompt=prompt, tokenizer=tokenizer, 
                                                        model=model, 
                                                        mutations_to_generate=len(original_feats[-1]), 
                                                        return_tokens=True, custom=True
                                                        ).detach().cpu().numpy()
        generated_feats.append(synt_gen[prompt_len:sample_len+prompt_len])
        # generated_feats.append(extract_sample_features(synt_gen[prompt_len:sample_len+prompt_len]))
        if len(original_feats[-1]) != len(generated_feats[-1]):
            print(f"Sample has different lengths: {len(original_feats[-1])} vs {len(generated_feats[-1])}")
            continue

    original_df = pd.DataFrame(original_feats, index=[f'original_{i+1}' for i in range(len(original_feats))])
    generated_df = pd.DataFrame(generated_feats, index=[f'synth_{i+1}' for i in range(len(generated_feats))])
    
    df = pd.concat([original_df, generated_df], ignore_index=False, sort=False)
    dists = pdist(df, metric=metric)
    # plot 1-dist to get similarity 
    similarity_matrix = pd.DataFrame(1.-squareform(dists), columns=df.index, index=df.index)

    # cmap = 'YlGnBu_r'
    sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu', 
                linewidths=.5, ax=ax, vmax=1.)

    plt.title(f'Sample Similarity ({metric})')
    if save_path:
        plt.savefig(f'{save_path}.pdf', format='pdf')
    plt.show()

    return dists

def plot_pca(original_samples, model, tokenizer, prompt_len=1, sample_len='all', 
            num_samples=10, figsize=(15,15), save_path=None):
    """
    Plot the similarity between original and generated samples.
    """
    original_feats = []
    generated_feats = []
    if sample_len == 'all':
        sample_len = min([len(s) for s in original_samples]) + 1
    else:
        assert isinstance(sample_len, int), "sample_len should be 'all' or an integer."
    
    if isinstance(num_samples, int):
        sample_idx = random.sample(range(len(original_samples)), num_samples)
    elif isinstance(num_samples, list):
        sample_idx = num_samples
    # print(f'Selected sample indices: {sample_idx}')

    for i in sample_idx:
        sample = original_samples[i]
        original_feats.append(tokenizer.encode(' '.join(sample))[:sample_len])
        # original_feats.append(extract_sample_features(sample))
        
        prompt = ' '.join(sample[:prompt_len])
        synt_gen = query_model(prompt=prompt, 
                                tokenizer=tokenizer, 
                                model=model, 
                                mutations_to_generate=len(original_feats[-1]), 
                                return_tokens=True, 
                                custom=True
                                ).detach().cpu().numpy()
        generated_feats.append(synt_gen[prompt_len:sample_len+prompt_len])
        # generated_feats.append(extract_sample_features(synt_gen[prompt_len:sample_len+prompt_len]))
        if len(original_feats[-1]) != len(generated_feats[-1]):
            print(f"Sample has different lengths: {len(original_feats[-1])} vs {len(generated_feats[-1])}")
            continue

    original_df = pd.DataFrame(original_feats, index=[f'original_{i+1}' for i in range(len(original_feats))])
    generated_df = pd.DataFrame(generated_feats, index=[f'synth_{i+1}' for i in range(len(generated_feats))])
    
    # df = pd.concat([original_df, generated_df], ignore_index=False, sort=False)
    # print(df.head(), df.index)

    pca = PCA(n_components=2)
    original_pca = pd.DataFrame(data = pca.fit_transform(original_df)
             , columns = ['principal component 1', 'principal component 2'])
    
    generated_pca = pd.DataFrame(data = pca.fit_transform(generated_df)
             , columns = ['principal component 1', 'principal component 2'])
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Principal Component 1',fontsize=16)
    plt.ylabel('Principal Component 2',fontsize=16)

    for target, color, label in zip([original_pca, generated_pca], 
                            ['cornflowerblue', 'yellowgreen'],
                            ['Original Samples', 'Generated Samples']):
        plt.scatter(target['principal component 1'], 
                    target['principal component 2'], 
                    c=color, s=100, alpha=0.8, label=label)

    plt.legend(loc='upper right', fontsize=12)
    plt.title(f'Generated vs Source Samples PCA', 
              fontdict={'fontsize': 18, 'weight': 'bold'})
    if save_path:
        plt.savefig(f'{save_path}.pdf', format='pdf')
    plt.show()

    return pca

################################ FORMATTING ################################ 
def extract_sample_features(sample_mutations):

        def _extend_feature_vector(features, field):
            """
            Extend the feature vector with new features
            """
            values = list(field.values())
            if isinstance(values[0], list):
                field_stats = sum(len(v) for v in values)
            else:
                field_stats = sum(values)
            if field_stats > 0:
                if isinstance(values[0], list):
                    features.extend(len(field[k])/field_stats for k in field.keys())
                else:
                    features.extend(field[k]/field_stats for k in field.keys())
            else:
                features.extend([0, 0, 0, 0])
            return features

        """
        Extract feature vector from mutation sequences
        """
        if isinstance(sample_mutations, str):
            sample_mutations, _, _ = get_valid_sequences([sample_mutations])
        
        features = []
        
        # Number of mutations
        features.append(len(sample_mutations))
        
        # Nucleotide composition
        nucleotides = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        mut_types = get_mutations_by_type(sample_mutations)
        gt_types = {'homozygous_ref':0,
                    'heterozygous':0,
                    'homozygous_alt':0}
        gt_lookup = {'0|0': 'homozygous_ref',
                    '0/0': 'homozygous_ref',
                    '1|0': 'heterozygous',
                    '0|1': 'heterozygous',
                    '1/0': 'heterozygous',
                    '0/1': 'heterozygous',
                    '1/1': 'homozygous_alt',
                    '1|1': 'homozygous_alt'}
        positions = []
        
        for mut in sample_mutations:
            _, pos, alleles = mut.split(':')
            alleles, gt = alleles.split('_')
            ref, alt = alleles.split('>')

            if '/' in ref:
                ref = ref.split('/')
            else:
                ref = [ref]
            for r in ref:
                # TODO: treat each multiallelic mut as one (NB: 
                # leads to different-shaped feature vectors)
                # if r not in nucleotides:
                #     nucleotides[r] = 0
                # nucleotides[r] += 1

                # instead treat each single nucleotide independently
                for i in range(len(r)):
                    nucleotides[r[i]] += 1

            if '/' in alt:
                alt = alt.split('/')
            else:
                alt = [alt]
            for a in alt:
                # if a not in nucleotides:
                #     nucleotides[a] = 0
                # nucleotides[a] += 1
                for i in range(len(a)):
                    nucleotides[a[i]] += 1

            positions.append(int(pos))
            gt_types[gt_lookup[gt]] += 1

        for field in [nucleotides, mut_types, gt_types]:
            # print(f"Field: {field}")
            features = _extend_feature_vector(features, field)

        features.extend([np.mean(positions),
                        np.std(positions) if len(positions) > 1 else 0,
                        min(positions),
                        max(positions)])
        
        return np.array(features)

def equalize_samples(samples_features):
    """
    Equalize the lengths of the generated and original samples by creating a larger
    vector with ALL mutations and setting the genotype of the sample as follows:
        - -1 if the sample doesn't have the mutation
        - 0 for homozygous ref
        - 1 for heterozygous
        - 2 for homozygous alt
    """
    raise NotImplementedError("This function is not implemented yet.") 

################################# DATA QC METRICS #################################  
import re
from data.data_utils import SEQUENCE_PATTERN, SNP_PATTERN, GT_PATTERN

def str_to_mutations_set(sequences, pattern=SEQUENCE_PATTERN):
    """
    From 'chr:pos_ref>alt chr:pos_ref>alt ...' to ['chr:pos_ref>alt', ...]
    """
    mutations_seqs = []
    for seq in sequences:
        muts = re.findall(pattern, seq)
        mutations_seqs.extend(muts)

    return mutations_seqs

def mutations_set_to_str(mutations_list):
    """
    From ['chr:pos_ref>alt', ...] to 'chr:pos_ref>alt chr:pos_ref>alt ...'
    """
    mutations_seqs = []
    for mut_set in mutations_list:
        mutations_seqs.append(' '.join(mut for mut in mut_set))

    return mutations_seqs

def get_valid_sequences(sequences, pattern=SEQUENCE_PATTERN, sample_wise=False, 
                        has_sample_names=False, include_sample_names=False):
    """
    Filter out invalid sequences from a list of sequences.
    
    Args:
        sequences (list): A list of sequences where each sequence is a list of 
                            mutations in the form 'chr:pos_ref>alt_a1|a2' for 
                            each sample.
        pattern (str): A regex pattern to match valid sequences.
    Returns:
        list (str): A list of valid (sub)sequences.
        list (float): A list of ratios of valid (sub)sequences vs total sequences.
    """
    total_sequences = len(sequences)
    valid_seq_ratio = [0 for _ in range(total_sequences)]
    
    all_sequences = []
    valid_sequences = []
    
    for i, seq in enumerate(sequences):
        all_muts_in_seq = [mut for mut in seq.split(' ') if mut != '']
        valid_muts_in_seq = []
        if has_sample_names:
            if include_sample_names:
                valid_muts_in_seq.append(all_muts_in_seq[0])  # Include the sample name
            all_muts_in_seq = all_muts_in_seq[1:]  # Skip the sample name if present

        valid_muts_in_seq.extend(re.findall(pattern, seq))
        valid_seq_ratio[i] = len(valid_muts_in_seq) / len(all_muts_in_seq)

        # if valid_seq_ratio[i] != 1.:
        #     print([mut for mut in all_muts_in_seq if mut not in valid_muts_in_seq])

        if sample_wise:
            valid_sequences.append(valid_muts_in_seq)
        else:
            valid_sequences.extend(valid_muts_in_seq)
        all_sequences.extend(mut for mut in all_muts_in_seq if mut != '')

    return valid_sequences, valid_seq_ratio, all_sequences

# ? duplicate ?
def get_novel_mutations(sequences, original_mutations, pattern=SEQUENCE_PATTERN, 
                        sample_wise=False, has_sample_names=False, 
                        include_sample_names=False):
    """
    """
    if sample_wise:
        all_original_muts = []
        existing_muts = []
        novel_muts = []
    else:
        all_original_muts = set()
        existing_muts = set()
        novel_muts = set()
    
    for i, (synth_seq, original_seq) in enumerate(zip(sequences, original_mutations)):
        original_seq_muts = original_seq.split(' ')
        original_seq_muts = [mut.split('_')[0] for mut in (original_seq_muts)]

        # get each mutation chr:pos:ref>alt without sample gt
        if isinstance(synth_seq, str):
            seq_mutations = synth_seq.split(' ')
        else:
            seq_mutations = synth_seq

        if has_sample_names:
            seq_mutations = seq_mutations[1:]  # Skip the sample name if present

        seq_mutations = [mut.split('_')[0] for mut in (synth_seq)]

        if sample_wise:
            all_original_muts.append([mut for mut in original_seq_muts])
            novel_muts.append([mut for mut in seq_mutations if mut not in original_seq_muts])
            existing_muts.append([mut for mut in seq_mutations if mut in original_seq_muts])
        else:
            all_original_muts.update([mut for mut in original_seq_muts])
            novel_muts.update([mut for mut in seq_mutations if mut not in original_seq_muts])
            existing_muts.update([mut for mut in seq_mutations if mut in original_seq_muts])

    return novel_muts, existing_muts, all_original_muts

def get_new_generated_mutations(sequences, original_sequences, verbose=False,
                                pattern=SNP_PATTERN, sample_wise=False):
    """
    Get unique mutations from a list of sequences.

    Args:
        sequences (list): A list of sequences where each sequence is a list of 
                            mutations in the form 'chr:pos_ref>alt_a1|a2' for 
                            each sample.
        pattern (str): A regex pattern to match valid sequences.
        
    Returns:
        set: A set of unique sequences.
    """
    original_unique_muts = set()
    if sample_wise:
        for seq in original_sequences:
            if isinstance(seq, str):
                valid_muts_in_seq = re.findall(pattern, seq)
            else:
                valid_muts_in_seq = seq
            original_unique_muts.update(mut for mut in valid_muts_in_seq)
    else:
        if isinstance(original_sequences, str):
            valid_muts_in_seq = re.findall(pattern, original_sequences)
        else:
            valid_muts_in_seq = original_sequences
        original_unique_muts.update(mut for mut in valid_muts_in_seq)
    
    if sample_wise:
        generated_unique_muts = [[] for _ in sequences]
        memorized_muts = [[] for _ in sequences]
        total_muts = [0 for _ in sequences]
        for i,seq in enumerate(sequences):
            if isinstance(seq, str):
                valid_muts_in_seq = re.findall(pattern, seq)
            else:
                valid_muts_in_seq = seq
            total_muts[i] += len(valid_muts_in_seq)

            for mut in valid_muts_in_seq:
                if mut not in original_unique_muts:
                    generated_unique_muts[i].append(mut)
                else:
                    memorized_muts[i].append(mut)
            if verbose:
                print(f"Total muts: {total_muts[i]} >> Unique: {len(generated_unique_muts[i])}, Memorized: {len(memorized_muts[i])}")
    else:
        generated_unique_muts = []
        memorized_muts = []
        total_muts = 0
        if isinstance(sequences, str):
            valid_muts_in_seq = re.findall(pattern, sequences)
        else:
            valid_muts_in_seq = sequences
        total_muts += len(set(valid_muts_in_seq))

        memorized_muts = set(valid_muts_in_seq).intersection(original_unique_muts)
        generated_unique_muts = set(valid_muts_in_seq).difference(original_unique_muts)
        # for mut in valid_muts_in_seq:
        #     if mut not in original_unique_muts:
        #         generated_unique_muts.append(mut)
        #     else:
        #         memorized_muts.append(mut)
        print(f"Total muts: {total_muts} >> Unique: {len(generated_unique_muts)}, Memorized: {len(memorized_muts)} out of {len(original_unique_muts)} original muts")
        if verbose:
            print(f"Total muts: {total_muts} >> Unique: {len(generated_unique_muts)}, Memorized: {len(memorized_muts)}")

    return generated_unique_muts, memorized_muts, total_muts

def get_locally_valid_mutations(sequences, pattern=SNP_PATTERN):
    """
    """
    locally_valid_muts = []
    for seq in sequences:
        if isinstance(seq, str):
            seq = re.findall(pattern, seq)
        for mut in seq:
            mut_chr, mut_pos, _ = mut.split(':')
            if mut and (0 <= int(mut_pos) <= CHR_LENGTHS[int(mut_chr)]):
                locally_valid_muts.append(mut)

    return locally_valid_muts

def get_uniqueness_score(samples_genotypes, pattern=SNP_PATTERN):
    """
    """
    from tqdm import tqdm
    unique_genotypes = [[] for _ in samples_genotypes]
    uniq_log = tqdm(samples_genotypes)
    for i,sample_seqs in enumerate(uniq_log):
        uniq_log.set_description(f"Processing sample {i+1}/{len(samples_genotypes)}")
        other_genotypes = set()
        for j,gts in enumerate(samples_genotypes):
            if j != i:
                other_genotypes.update(gts)
        unique_genotypes[i] = [mut for mut in sample_seqs if mut not in other_genotypes]
        # print(f"Sample {i+1} has {len(unique_genotypes[i])} unique genotypes out of {len(samples_genotypes[i])} total genotypes.")
    unique_sequences_ratio = [len(unique_genotypes[i]) / len(samples_genotypes[i]) for i in range(len(samples_genotypes))]
    return unique_genotypes, unique_sequences_ratio

################################# UTILITY METRICS #################################    
import allel

def get_allelic_distribution(genotypes, is_synthetic=False):
    """
    Get the distribution of alleles in the provided genotypes.
    """
    # all samples have same mutation, but different genotypes
    if isinstance(genotypes[0], list):
        mutations_list = set(genotypes[0])
    elif not is_synthetic:
        mutations_list = set(re.findall(SNP_PATTERN, genotypes[0]))
    else:
        sample_mutations = [re.findall(SNP_PATTERN, sample) for sample in genotypes]
        mutations_list = set()
        for mut in sample_mutations:
            mutations_list.update(mut)

    allele_counts = {'biallelic':0,
                    'biallelic_refs':0,
                    'biallelic_alts':0,
                    'multiallelic_refs':0,
                    'multiallelic_alts':0}
    
    for mut in mutations_list:
        _,_,alleles = mut.split(':')
        alleles, gt = alleles.split('_')
        ref, alt = alleles.split('>')
        if not ('/' in ref or '/' in alt):
            allele_counts['biallelic'] += 1
        elif '/' in ref:
            num_refs = len(ref.split('/'))
            if num_refs > 2:
                allele_counts['multiallelic_refs'] += 1
            else:
                allele_counts['biallelic_refs'] += 1
        elif '/' in alt:
            num_alts = len(alt.split('/'))
            if num_alts > 2:
                allele_counts['multiallelic_alts'] += 1
            else:
                allele_counts['biallelic_alts'] += 1

    return allele_counts

def get_allele_length_distribution(genotypes, is_synthetic=False):
    """
    Get the distribution of allele lengths in the provided genotypes.
    """
    ref_allele_lengths = {i:0 for i in range(1, 31)}
    alt_allele_lengths = {i:0 for i in range(1, 31)}
    if isinstance(genotypes[0], list):
        mutations_list = set(genotypes[0])
    elif not is_synthetic:
        mutations_list = set(re.findall(SNP_PATTERN, genotypes[0]))
    else:
        sample_mutations = [re.findall(SNP_PATTERN, sample) for sample in genotypes]
        mutations_list = set()
        for mut in sample_mutations:
            mutations_list.update(mut)
    
    for mut in mutations_list:
        _,_,alleles = mut.split(':')
        alleles, gt = alleles.split('_')
        ref, alt = alleles.split('>')
        if '/' in ref:
            for r in ref.split('/'):
                if len(r) not in ref_allele_lengths.keys():
                    ref_allele_lengths[len(r)] = 0
                ref_allele_lengths[len(r)] += 1
        else:
            if len(ref) not in ref_allele_lengths.keys():
                ref_allele_lengths[len(ref)] = 0
            ref_allele_lengths[len(ref)] += 1
        
        if '/' in alt:
            for a in alt.split('/'):
                if len(a) not in alt_allele_lengths.keys():
                    alt_allele_lengths[len(a)] = 0
                alt_allele_lengths[len(a)] += 1
        else:
            if len(alt) not in alt_allele_lengths.keys():
                alt_allele_lengths[len(alt)] = 0
            alt_allele_lengths[len(alt)] += 1

    return ref_allele_lengths, alt_allele_lengths

def get_sequences_by_chromosome(data, chr=DEFAULT_CHR, sample_wise=False):
    """
    Returns a subset of the input data containing only the sequences for the specified chromosome.
    
    Args:
        data (list): list of mutations in the form 'chr:pos_ref>alt_a1|a2' for each sample.
        chr (int): Chromosome number to filter by (defaults to DEFAULT_CHR).
    
    Returns:
        list: Filtered list with sequences for the specified chromosome.
    """
    assert isinstance(chr, int) and (0 < chr < 23), \
        "Chromosome must be an integer between 1 and 22."

    if sample_wise:
        return [[seq for seq in sample if seq.startswith(f"{chr}:")] for sample in data]
    else:
        return [seq for seq in data if seq.startswith(f"{chr}:")]

def get_mutations_by_type(sequences, pattern=SNP_PATTERN, **kwargs):
    """
    Split mutations into different categories based on their types.
    Currently supports:
        - deletions: one or more reference alleles deleted at a position
            EG: '1:12345678:ACT>CT' (deletion of 'A' at position 12345678)
        - insertions: one or more alternate alleles inserted at a position
            EG: '1:12345678:CT>CAT' (insertion of 'A' at position 12345678)
        - substitutions: one or more reference alleles substituted with one or more alternate alleles at a position
            EG: '1:12345678:ACT>GCT' (substitution of 'A' with 'G' at position 12345678)
        - biallelic: one reference and one alternate allele at a position
            EG: '1:12345678:T>G' (biallelic substitution at position 12345678)
        - multiallelic mutations: one reference allele and multiple (2+) alternate alleles at a position
            EG: '1:12345678:GCT/AAT/TAG>ACT' (multiallelic mutation at position 12345678)
    """

    mutation_types = {'deletions': [], 
                      'insertions': [], 
                      'substitutions': [],
                      'biallelic': [],
                      'multiallelic': []}
    seen_muts = set()
    
    for seq in sequences: # for each sample
        if isinstance(seq, str):
            muts = re.findall(pattern, seq)
        else:
            muts = seq
        
        for mut in muts:
            curr_categories = set()

            chr, pos, alleles = mut.split(':')
            refs, alts = alleles.split('>')
            if f'{chr}:{pos}:{refs}>{alts}' not in seen_muts:
                seen_muts.add(f'{chr}:{pos}:{refs}>{alts}')
                # Get allele counts (bi- vs multi-allelic)
                if not ('/' in alts or '/' in refs):
                    # If there are no slashes, it is a biallelic mutation
                    curr_categories.add('biallelic')
                else:
                    if '/' in alts:
                        alts = alts.split('/')
                    if '/' in refs:
                        refs = refs.split('/')
                    curr_categories.add('multiallelic')

                # Get mutation type (deletion, insertion, substitution)
                # A. biallelic mutations
                if isinstance(refs, str) and isinstance(alts, str):
                    if len(refs) > len(alts):
                        curr_categories.add('deletions')
                    elif len(refs) < len(alts):
                        curr_categories.add('insertions')
                    else:
                        curr_categories.add('substitutions')
                # B. multiallelic mutations: different refs/alts can lead to 
                #   multiple categories for the same mutation locus
                else:
                    if isinstance(refs, list):
                        for ref in refs:
                            if len(ref) > len(alts):
                                curr_categories.add('deletions')
                            elif len(ref) < len(alts):
                                curr_categories.add('insertions')
                            else:
                                curr_categories.add('substitutions')
                    if isinstance(alts, list):
                        for alt in alts:
                            if len(refs) > len(alt):
                                curr_categories.add('deletions')
                            elif len(refs) < len(alt):
                                curr_categories.add('insertions')
                            else:
                                curr_categories.add('substitutions')

                for category in curr_categories:
                    mutation_types[category].append(mut)
    return mutation_types

################################# PRIVACY METRICS #################################  

def get_high_risk_samples(sample_genotypes,
                        rare_muts_file='data/sources/rare_snps.txt',
                        return_rare_gts_only=False):
    """
    """
    rare_muts = open(rare_muts_file, 'r').readlines()
    rare_muts = [mut.rstrip('\n') for mut in rare_muts]

    high_risk_samples = {}
    for sample, genotypes in sample_genotypes.items():
        if isinstance(genotypes, dict):
            # handle source data format
            genotypes = genotypes['genotypes']
        genotypes = get_valid_sequences([genotypes])
        high_risk_gts = []
        for mut in genotypes:
            # return samples that have non-homozygous ref genotype for rare variants
            if mut in rare_muts and mut.split('_') != '0|0':
                if return_rare_gts_only:
                    high_risk_gts.append(mut)
                else:
                    high_risk_gts = genotypes
                    break
        high_risk_samples[sample] = high_risk_gts
    return high_risk_samples

def get_memorization_score(sequences, model, tokenizer, return_avg=True, query=False, 
                           generated_sequences=None, seq_length=100, return_tokens=False):    
    """
    Calculate the memorization score of a model on a set of sequences by comparing how
    many generated mutations (including genotype) match the original data.
    
    Args:
        sequences (list): A list of sequences where each sequence is a list of 
                        mutations in the form 'chr:pos_ref>alt_a1|a2' for each sample.
        model: The trained model to evaluate.
        tokenizer: The tokenizer used to encode the sequences.
        
    Returns:
        list: the sample-wise memorized generated mutations
        (list or float): The memorization score (either sample-wise or average). 
                        Represents the fraction of memorized mutations to total 
                        generated mutations on a sample-basis.
    """
    assert generated_sequences is not None or query,\
        "Either generated_sequences or query must be provided."
    total_original_muts = []
    all_generated_muts = []
    memorized_muts = []
    for i,seq in enumerate(sequences):
        if query:
            if isinstance(seq, list):
                seq = ' '.join(seq)
            generated_seq = query_model(seq, tokenizer, model, 
                                        mutations_to_generate=seq_length,
                                        return_tokens=return_tokens)
        else:
            generated_seq = generated_sequences[i]
            # if more than one sample generated per prompt
            if isinstance(generated_seq, list):
                generated_seq = [''.join(tokenizer.decode(s)) for s in generated_seq]
            elif not isinstance(generated_seq, str):
                generated_seq = ' '.join(tokenizer.decode(generated_seq))
        
        input_muts = re.findall(SEQUENCE_PATTERN, seq) if isinstance(seq, str) else seq

        if isinstance(generated_seq, list): # if more than one sample generated per prompt
            for g_seq in generated_seq:
                total_original_muts.append(input_muts)

                generated_muts = re.findall(SEQUENCE_PATTERN, g_seq)  if isinstance(g_seq, str) else g_seq
                all_generated_muts.append([mut for mut in generated_muts])
                overlapping_muts = set(input_muts).intersection(generated_muts)
                memorized_muts.append([mut for mut in overlapping_muts])

    memorization_score = [len(mem)/len(gen) for mem, gen in zip(memorized_muts, all_generated_muts)]
    if return_avg:
        # Return the average memorization score for all sequences
        return memorized_muts, np.mean(memorization_score)
    else:
        return memorized_muts, memorization_score
    