import torch
import tqdm
import re
import numpy as np
import warnings
import torch.nn.functional as F

from sklearn.metrics import pairwise_distances
from data.data_utils import corpus_to_VCF, get_running_device, set_seed
from metrics import *
from models.tokenizers import RegexTokenizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from models.MinGPT import MinGPT
from data.dataset import GPT_SPECIAL_TOKENS

def prompt_based_information_leakage(target_model, tokenizer, prompting_data,
                                distance_metric='cosine', **kwargs):
    prompt_types = list(prompting_data.keys())
    include_prompt_len = kwargs.get('include_prompt_len', False)
    extract_features = kwargs.get('extract_features', False)

    print(f"Extracting {'feature-based' if extract_features else 'token-based'} distances using {distance_metric} metric...")

    generated_samples = {k:[] for k in prompt_types}
    distances = {k:[] for k in prompt_types}
    return_avg = kwargs.get('return_avg', True)

    generation_loop = tqdm.tqdm(prompting_data.items()) # {prompt_type: {prompt, sample_gts}}
    for prompt_type, prompt_data in generation_loop:

        prompts = prompt_data['prompts'] # list
        samples = prompt_data['samples'] # list

        if include_prompt_len:
            desc = f"Prompting for {prompt_type} data (len={len(prompts[0].split(' '))})"
        else:
            desc = f"Prompting for {prompt_type} data"
        generation_loop.set_description(desc=desc)

        for prompt, original_sample in zip(prompts, samples):
            # print("Prompting with", prompt)
            if isinstance(tokenizer, RegexTokenizer):
                tokenized_prompt = tokenizer.encode(prompt, allowed_special='all')
                tokenized_prompt = torch.LongTensor(tokenized_prompt).unsqueeze(0)
            else:
                tokenized_prompt = torch.LongTensor([tokenizer.encode(prompt)])

            max_sample_length = kwargs.get('max_sample_length', 500) # len(original_sample)

            prompt_generated_samples = []
            distance = []
            for i in range(kwargs.get('rounds', 1)):
                # print('Generating sample', i+1)
                generation_loop.set_postfix_str(f'Generating sample {i+1}')
                model_sample = query_model(prompt=prompt, tokenizer=tokenizer, 
                                            model=target_model, 
                                            mutations_to_generate=max_sample_length, 
                                            return_tokens=(not extract_features), 
                                            **kwargs)
                if not extract_features:
                    model_sample = model_sample.detach().cpu().numpy()
                
                prompt_generated_samples.append(model_sample)
                if extract_features:
                    sample_features = extract_mutation_features(model_sample[len(prompt):])
                    decoded_original = tokenizer.decode(original_sample[len(prompt):])
                    original_features = extract_mutation_features(decoded_original)
                    sample_dist = pairwise_distances(np.array(sample_features).reshape(1, -1), 
                                        np.array(original_features).reshape(1, -1), 
                                        metric=distance_metric).tolist()
                else:
                    prompt_len = len(tokenized_prompt)
                    sample_dist = pairwise_distances(model_sample[prompt_len:max_sample_length].reshape(1, -1), 
                                        np.array(original_sample[prompt_len:max_sample_length]).reshape(1, -1), 
                                        metric=distance_metric).tolist()
                # distance.extend(sample_dist[0])
                distance.append(sample_dist[0][0])
            # print("\t", prompt_type, "distance:", distance)
            generated_samples[prompt_type].append(prompt_generated_samples)
            if return_avg:
                # mean over rounds
                distances[prompt_type].append(np.mean(distance))
            else:
                distances[prompt_type].append(distance)

    return generated_samples, distances

def nearest_neighbor_matching(target_model, tokenizer, prompts, 
                            original_samples, distance_metric='cosine', **kwargs):
    """
    Perform nearest neighbor matching between original and generated data.
    """
    all_distances = [[] for _ in range(len(prompts))] # [S,N], S=number of generated samples, N=number of original samples
    nn_distances = {'original_id': [], 'distance': []} # [1, S]
    min_original_sample_len = min([len(s) for s in original_samples])

    generation_loop = tqdm.tqdm(prompts)
    for i, prompt in enumerate(generation_loop):
        generation_loop.set_description(desc=f"Prompting sample {i+1}/{len(prompts)}")
        print("Prompting with", prompt)
        max_sample_length = kwargs.get('max_sample_length', min_original_sample_len)

        generated_sample = query_model(prompt=prompt,
                                    tokenizer=tokenizer, 
                                    model=target_model, 
                                    mutations_to_generate=max_sample_length, 
                                    return_tokens=True, custom=True
                                    ).detach().cpu().numpy()[0]
            
        for j, og_sample in enumerate(original_samples):
            print(f"        Comparing with original sample {j+1}")
            dist = pairwise_distances(generated_sample[:max_sample_length].reshape(1, -1), 
                                    np.array(og_sample[:max_sample_length]).reshape(1, -1), 
                                    metric=distance_metric).tolist()
            all_distances[i].append(dist[0][0])

        nn_distances['original_id'].append(np.argmin(all_distances[i]))
        nn_distances['distance'].append(min(all_distances[i]))

        generation_loop.set_postfix_str(f"Closest neighbour {nn_distances['original_id'][-1]} for sample {i+1}: {nn_distances['distance'][-1]}")

    return nn_distances, all_distances

def get_sample_memorization_ratio(sequence, training_sequences, **kwargs):
    from data.data_utils import MUTATION_PATTERN, SEQUENCE_PATTERN

    pattern = kwargs.get('pattern', SEQUENCE_PATTERN)

    memorization_ratio = []
    generated_mutations = [re.findall(pattern, seq) for seq in sequence]
    original_mutations = [re.findall(pattern, seq) for seq in training_sequences]

    memorization_log = tqdm.tqdm(generated_mutations)
    for j, synth_sample in enumerate(memorization_log):
        memorization_log.set_description(desc=f"Calculating memorization for sample {j+1}/{len(generated_mutations)}")
        sample_memo = {'sample': [],
                        'memorization_ratio': []}
        for i, og_sample in enumerate(original_mutations):
            common_muts = set(synth_sample).intersection(set(og_sample))
            if common_muts != []:
                sample_memo['sample'].append((j, i))
                sample_memo['memorization_ratio'].append(len(common_muts)/len(synth_sample))
        
        memorization_ratio.append(max(sample_memo['memorization_ratio']) if sample_memo['memorization_ratio'] else 0)

        idx = sample_memo['sample'][-1]
        memorization_log.set_postfix_str(f'Sample {idx[0]}: {memorization_ratio[-1]} (Original sample {idx[1]})')

    return memorization_ratio

def extract_mutation_features(sequence):
    from data.data_utils import SEQUENCE_PATTERN
    """Extract mutation-specific features for each sample"""
    mutation_stats = {
        'mutation_rate': 0,
        'allele_mutation_rate': 0,
        'deletions': 0,
        'insertions': 0,
        'substitutions': 0,
        'biallelic': 0,
        'multiallelic': 0,
        'homozygous_ref': 0,
        'homozygous_alt': 0,
        'heterozygous_ref': 0}

    try:
        if '<MUT_SEP>' in sequence:
            sequence = sequence.replace('<MUT_SEP>', ' ').strip()

        if isinstance(sequence, str):
            muts = re.findall(SEQUENCE_PATTERN, sequence)
        elif isinstance(sequence, list):
            muts = sequence

        mutations_per_sample = 0
        allele_mutation_rate = {'mutated_alleles': 0, 'total_alleles': 0}
        for i,mut in enumerate(muts):
            _, _, alleles = mut.split(':')
            alleles, gt = alleles.split('_')
            if gt not in ['0|0', '0/0']: # only get statistics for mutated genotypes
                allele_mutation_rate['total_alleles'] += 2
                mutations_per_sample += 1
                if '|' in gt:
                    allele1, allele2 = gt.split('|')
                elif '/' in gt:
                    allele1, allele2 = gt.split('/')

                # get genotype counts
                if int(allele1) == 0 or int(allele2) == 0:
                    mutation_stats['heterozygous_ref'] += 1
                    allele_mutation_rate['mutated_alleles'] += 1
                else:
                    mutation_stats['homozygous_alt'] += 1
                    allele_mutation_rate['mutated_alleles'] += 2


                mutation_stats['allele_mutation_rate'] = (allele_mutation_rate['mutated_alleles'] / allele_mutation_rate['total_alleles']) if allele_mutation_rate['total_alleles'] > 0 else 0
                mutation_stats['mutation_rate'] = (mutations_per_sample / len(muts)) if len(muts) > 0 else 0

                # Get allele counts (bi- vs multi-allelic)
                refs, alts = alleles.split('>')

                if not '/' in alts and not '/' in refs:
                    # If there are no slashes, it is a biallelic mutation
                    mutation_stats['biallelic'] += 1
                else:
                    if '/' in alts:
                        alts = alts.split('/')
                    if '/' in refs:
                        refs = refs.split('/')
                    mutation_stats['multiallelic'] += 1

                # Get mutation type (deletion, insertion, substitution)
                # A. biallelic mutations
                if isinstance(refs, str) and isinstance(alts, str):
                    if len(refs) > len(alts):
                        mutation_stats['deletions'] += 1
                    elif len(refs) < len(alts):
                        mutation_stats['insertions'] += 1
                    else:
                        mutation_stats['substitutions'] += 1
                # B. multiallelic mutations: different refs/alts can lead to
                #   multiple categories for the same mutation locus
                else:
                    if isinstance(refs, list):
                        for ref in refs:
                            if len(ref) > len(alts):
                                mutation_stats['deletions'] += 1
                            elif len(ref) < len(alts):
                                mutation_stats['insertions'] += 1   
                            else:
                                mutation_stats['substitutions'] += 1
                    if isinstance(alts, list):
                        for alt in alts:
                            if len(refs) > len(alt):
                                mutation_stats['deletions'] += 1
                            elif len(refs) < len(alt):
                                mutation_stats['insertions'] += 1
                            else:
                                mutation_stats['substitutions'] += 1
            else:
                mutation_stats['homozygous_ref'] += 1
    except Exception as e:
        print(f"Error parsing mutation features for sequence {i+1}: {e}")
        pass

    return list(mutation_stats.values())

def extract_features(sequences, model, tokenizer, **kwargs):

    def get_model_sequence(prompt, model, tokenizer, max_length=100, **kwargs):
        tokens_to_generate = max_length # * 10
        device = kwargs.get('device', 'cpu')
        
        model = model.to(device)
        model.eval()
        if isinstance(prompt, str):
            if isinstance(tokenizer, RegexTokenizer):
                tokenized_prompt = tokenizer.encode(prompt, allowed_special='all')
                tokenized_prompt = torch.LongTensor(tokenized_prompt).unsqueeze(0)
            else:
                tokenized_prompt = torch.LongTensor([tokenizer.encode(prompt)])
        else:
            tokenized_prompt = prompt

        # print("Prompting with", tokenized_prompt)
        tokenized_prompt = tokenized_prompt.to(device)
        if isinstance(model, MinGPT):
            generated_tokens = model.generate(tokenized_prompt, tokens_to_generate)
        else:
            pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256
            eos_tok = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
            # print(f"Using pad token {pad_tok}, eos token {e
            generated_tokens = model.generate(input_ids.to(device),
                                max_new_tokens=tokens_to_generate,
                                num_return_sequences=1,
                                temperature=0.8,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=pad_tok,
                                eos_token_id=eos_tok)

            # attention_mask = torch.ones_like(tokenized_prompt).to(device)
            # generated_tokens = model.generate(tokenized_prompt, max_new_tokens=tokens_to_generate, attention_mask=attention_mask)
        model_sample = tokenizer.decode(generated_tokens[0].tolist())

        # model_sample = query_model(prompt=prompt, tokenizer=tokenizer, 
        #                             model=model, 
        #                             mutations_to_generate=max_length, 
        #                             return_tokens=False, custom=True)

        return model_sample

    """
    Adapted from: https://huggingface.co/docs/transformers/en/perplexity
    """
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    include_genomic_features = kwargs.get('include_genomic_features', True)
    features = []
    prompt_length = kwargs.get('prompt_length', 2)

    context_window = kwargs.get('context_window', 1024) # 512
    max_length = kwargs.get('max_length', None)

    default_features = [100.0, 0.001, 0.0, 0.0, 0.0]
    if include_genomic_features:
        default_features += [0.0]*10
                
    with torch.no_grad():
        model.eval()
        extraction_log = tqdm.tqdm(sequences,
                                desc=f"Extracting {'hybrid ' if include_genomic_features else 'model-based '}features from {len(sequences)} samples")

        for i, sequence in enumerate(extraction_log):

            nll_sum = 0.0
            confidence_sum = 0.0
            logits_magnitude_sum = 0.0
            n_tokens = 0
            prev_end_loc = 0
            all_losses = []

            try:
                if isinstance(model, MinGPT):
                    tokens = torch.LongTensor(tokenizer.encode(sequence, 
                                            allowed_special='all')).unsqueeze(0)
                    seq_len = tokens.size(1)
                    max_model_length = model.config.block_size
                else:
                    tokens = tokenizer(sequence, return_tensors='pt')
                    seq_len = tokens.input_ids.size(1)
                    max_model_length = model.config.n_positions

                for begin_loc in range(0, seq_len, context_window):
                    end_loc = min(begin_loc + max_model_length, seq_len)
                    trg_len = end_loc - prev_end_loc 
                    extraction_log.set_postfix_str(f"Processing tokens ({begin_loc} to {end_loc})")
                    
                    if isinstance(model, MinGPT):
                        input_ids = tokens[:, begin_loc:end_loc].to(device)
                        target_ids = input_ids.clone()
                        target_ids[:, :-trg_len] = -100
                        
                        logits, loss = model(input_ids, target_ids)
                    else:
                        input_ids = tokens.input_ids[:, begin_loc:end_loc].to(device)
                        target_ids = input_ids.clone()
                        target_ids[:, :-trg_len] = -100

                        outputs = model(input_ids, labels=target_ids, loss_type='ForCausalLMLoss')
                        logits = outputs.logits
                        loss = outputs.loss
                    
                    all_losses.append(loss.item())

                    # Accumulate the total negative log-likelihood and the total number of tokens
                    num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
                    # print(num_valid_tokens)
                    batch_size = target_ids.size(0)
                    num_loss_tokens = num_valid_tokens - batch_size  # account for shift

                    if num_loss_tokens > 0: 
                        nll_sum += loss * num_loss_tokens
                        
                        # confidence (average max softmax probability)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = target_ids[..., 1:].contiguous()
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)
                        valid_mask = shift_labels != -100
                        
                        probs = F.softmax(shift_logits, dim=-1)
                        max_probs = torch.max(probs, dim=-1)[0]
                        valid_probs = max_probs[valid_mask]
                        confidence_sum += valid_probs.sum().item()
                        
                        # logits magnitude (L2 norm) for each token's logits
                        logits_norms = torch.norm(shift_logits, p=2, dim=-1)  # L2 norm across vocab dimension
                        valid_norms = logits_norms[valid_mask]
                        logits_magnitude_sum += valid_norms.sum().item()
                        
                        n_tokens += num_loss_tokens
                    
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

                # Calculate final metrics
                extraction_log.set_postfix_str(f"Calculating Metrics")
                if n_tokens > 0:
                    avg_loss = nll_sum / n_tokens
                    loss_variance = np.var(all_losses)
                    avg_confidence = confidence_sum / n_tokens
                    avg_logits_magnitude = logits_magnitude_sum / n_tokens
                    perplexity = torch.exp(avg_loss).item()
                    avg_loss = avg_loss.item()

                else:
                    avg_loss = float('inf')
                    loss_variance = 0.0
                    avg_confidence = 0.0
                    avg_logits_magnitude = 0.0
                    perplexity = float('inf')

                seq_features = [perplexity, avg_confidence, loss_variance,
                                avg_logits_magnitude, avg_loss]
                
                if include_genomic_features:
                    # Mutation features
                    prompt = ' '.join(sequence.split(" ")[:prompt_length])
                    extraction_log.set_postfix_str(f"Getting Mutation Features from prompt {(prompt)}")
                    generated_sequences = get_model_sequence(prompt, model, tokenizer, **kwargs)
                    mutation_features = extract_mutation_features(generated_sequences)
                    # mutation_features = extract_mutation_features(sequence)
                    seq_features += mutation_features

                features.append(seq_features)                    
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                features.append(default_features)

    return np.array(features)

def analyze_feature_importance(member_features, non_member_features):
    """
    """
    # Prepare data
    X = np.vstack([member_features, non_member_features])
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy')
    rf.fit(X, y)
    
    feature_names = ['Perplexity', 'Average Confidence', 'Loss Variance', 
                    'Avg Logit Magnitude', 'Average Loss', 
                    'Mutation Rate', 'Allele Mutation Rate',
                    'Deletions', 'Insertions', 'Substitutions', 'Biallelic', 
                    'Multiallelic', 'Homozygous Ref', 'Homozygous Alt', 
                    'Heterozygous Ref']

    importance_dict = dict(zip(feature_names, rf.feature_importances_))
    
    return importance_dict

def threshold_attack(member_features, non_member_features, feature_idx=1, **kwargs):
    """
    """
    # Use loss feature by default
    member_losses = member_features[:, feature_idx]
    non_member_losses = non_member_features[:, feature_idx]
    
    # Find optimal threshold
    all_losses = np.concatenate([member_losses, non_member_losses])
    thresholds = np.percentile(all_losses, np.linspace(0, 100, 101))
    
    best_accuracy = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        # Predict membership (lower loss = more likely to be member)
        member_pred = (member_losses <= threshold).astype(int)
        non_member_pred = (non_member_losses <= threshold).astype(int)
        
        accuracy = (np.sum(member_pred) + np.sum(1 - non_member_pred)) / (len(member_pred) + len(non_member_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Final predictions with best threshold
    member_pred = (member_losses <= best_threshold).astype(int)
    non_member_pred = (non_member_losses <= best_threshold).astype(int)
    
    # Calculate metrics
    y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    y_pred = np.concatenate([member_pred, non_member_pred])
    y_scores = np.concatenate([1 - member_losses/np.max(all_losses), 
                                1 - non_member_losses/np.max(all_losses)])
    
    auc = roc_auc_score(y_true, y_scores)

    avg = kwargs.get('average', 'macro')
    prec, rec, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
    
    return {'auc': auc,
            'accuracy': best_accuracy,
            'precision': prec,
            'recall': rec,
            'fscore': fscore,
            'threshold': best_threshold,
            'attack_advantage': auc - 0.5}

def ml_based_attack(member_features, non_member_features,
                    attack_type='logistic', **kwargs):

    """
    """
    # Prepare data
    X = np.vstack([member_features, non_member_features])
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
    
    # Split for attack model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
    
    # Choose attack model
    if attack_type == 'logistic':
        max_iter = kwargs.get('max_iter', 1000)
        attack_model = LogisticRegression(max_iter=max_iter)
    elif attack_type == 'rf':
        n_estimators = kwargs.get('n_estimators', 100)
        attack_model = RandomForestClassifier(n_estimators=n_estimators)
    else:
        import math
        if attack_type != 'knn':
            warnings.warn(f"Unknown attack type {attack_type}, defaulting to KNN.")
        n_neighbors = kwargs.get('n_neighbors', int(math.sqrt(len(X_train))//2)) # sqrt(num_samples)/2
        attack_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train attack model
    attack_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = attack_model.predict(X_test)
    y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    avg = kwargs.get('average', 'macro')
    prec, rec, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)

    return {'auc': auc,
            'accuracy': accuracy,
            'precision': prec,
            'recall': rec,
            'fscore': fscore,
            'attack_advantage': auc - 0.5}

def run_all_mia_attacks(model, tokenizer, train_samples, 
                                holdout_samples, **kwargs):
    """
    """
    set_seed(kwargs.get('seed', 42))

    max_length = kwargs.get('max_length', 100)
    include_genomic_features = kwargs.get('include_genomic_features', True)
    member_features = extract_features(train_samples, model, tokenizer, 
                                    max_length=max_length, 
                                    include_genomic_features=include_genomic_features)
    non_member_features = extract_features(holdout_samples, model, tokenizer, 
                                    max_length=max_length, 
                                    include_genomic_features=include_genomic_features)

    results = {}
    
    # Threshold attack
    tqdm.tqdm.write("Running threshold attack...")
    results['threshold'] = threshold_attack(member_features,
                                            non_member_features,
                                            **kwargs)
    
    # ML-based attacks
    ml_attacks = ['logistic', 'rf', 'knn']
    attacks_log = tqdm.tqdm(ml_attacks)
    for attack_type in attacks_log:
        try:
            attacks_log.set_description(f"Running {attack_type} attack...")
            results[attack_type] = ml_based_attack(member_features,
                                                non_member_features,
                                                attack_type,
                                                **kwargs)
        except Exception as e:
            attacks_log.write(f"Error in {attack_type} attack: {e}")
            results[attack_type] = {'error': str(e)}
    
    return results, member_features, non_member_features

def privacy_risk_assessment(results):
    """
    """
    max_auc = max([result.get('auc', 0) for result in results.values() if 'auc' in result])
    
    if max_auc > 0.8:
        risk_level = "HIGH"
    elif max_auc > 0.65:
        risk_level = "MODERATE"
    elif max_auc > 0.55:
        risk_level = "LOW-MODERATE"
    else:
        risk_level = "LOW"
    
    desc = f"Attacks Results: Risk Level: {risk_level} (Max AUC={max_auc:.4f})\n"
    desc += "================================================================\n"
    for attack, result in results.items():
        if 'auc' in result:
            desc += f"- {attack.upper()}: AUC={result['auc']:.4f}, Accuracy={result['accuracy']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, F-Score={result['fscore']:.4f}\n"

    return desc

def plot_mia_results(results, **kwargs):
    results_to_plot = [[
        mia_results['auc'],
        mia_results['accuracy'],
        mia_results['precision'],
        mia_results['recall'],
        mia_results['fscore'],
        mia_results['attack_advantage']
    ] for mia_results in results.values()]
    plot_metrics_by_model(results_to_plot, model_names=['AUC', 'Accuracy', 'Precision', 
                                                      'Recall', 'F1 Score', 'Attack Advantage'], 
                                        labels = list(results.keys()),
                                        x_ticks=['AUC', 'Accuracy', 'Precision', 
                                                  'Recall', 'F1 Score', 'Attack Advantage'],
                                        xlabel = 'Attack', ylabel='Score',
                                        legend_title='MIA metric',
                                        title=kwargs.get('title', "MIA Results"), 
                                        save_path=kwargs.get('save_path', None),
                                        bbox_to_anchor=kwargs.get('bbox_to_anchor', (0.8, 0.9)))

def plot_mia_comparison(results, results_dp, **kwargs):
    for ((res_k, res), (dp_res_k, dp_res)) in zip(results.items(), 
                                                    results_dp.items()):
        results_to_plot = [[
            res['auc'],
            res['accuracy'],
            res['precision'],
            res['recall'],
            res['fscore'],
            res['attack_advantage']],
            [dp_res['auc'],
            dp_res['accuracy'],
            dp_res['precision'],
            dp_res['recall'],
            dp_res['fscore'],
            dp_res['attack_advantage']]]
        plot_metrics_by_model(results_to_plot, 
                            model_names=['AUC', 'Accuracy', 'Precision', 
                                    'Recall', 'F1 Score', 'Attack Advantage'], 
                            labels = ['minGPT', f'minGPT DP (ε={kwargs.get("eps", "?")})'],
                            x_ticks=['AUC', 'Accuracy', 'Precision', 
                                    'Recall', 'F1 Score', 'Attack Advantage'],
                            xlabel = 'Metric', ylabel='Score',
                            title=f'MIA {res_k} Results (Model Features)', 
                            cmap='tab20b',
                            save_path=kwargs.get('save_path', None),
                            bbox_to_anchor=(0.8, 0.9))

def run_blackbox_mia(model, tokenizer, train_sequences, test_sequences, **kwargs):
        """
        """
        
        results, member_features, non_member_features = run_all_mia_attacks(model, tokenizer, train_sequences, test_sequences, **kwargs)

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