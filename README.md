# Privacy-Preserving Synthetic Genomes

This repository contains the implementation and experimental code for my MSc Advanced Computing Dissertation on "**Differentially-Private Language Models for the Generation of Synthetic Genetic Mutation Profiles**".
The project explores the critical need for privacy-preserving approaches tailored for genomic data, while evaluating the utility of genetic information for realism preservation.

## Background

The public sharing of data, methodologies and discoveries within the scientific community, is one of the main factors that has, over centuries, enabled large-scale and international scientific collaborations to achieve formidable research results. A prime example comes from the incredible amounts of genomic data generated as biproduct of the Big Biological Data era, which has led to outstanding discoveries in personalized medicine and biomedical research.


Such unprecedented availability of sensitive data, however, quickly raised new concerns around Genomic Privacy and what it entails for the public sharing of genomes, spreading worry amongst the scientific community. In the past decade, synthetic data has come forward as a promising solution to this ethical and legal dilemma, as it could mitigate the legal issues surrounding sensitive data sharing by eliminating the exposure of real individuals’ information.


## Aims and Results

In this research, we propose a sample-level **Synthetic Genetic Variant** data generation method via **Differentially Private Language Models** (LM), which applies the mathematically proven privacy guarantees of **Differential Privacy** (DP) to sample-level genomics. We show that both fine-tuned and custom-trained LMs are viable mock genomic variant generators, and that DP-augmented training successfully leads to a decrease in adversarial attack success. Furthermore, we showed that smaller generative models naturally offer, on average, more robust privacy guarantees compared to larger models without significant decrease in utility.


Additionally, we introduce a tailored privacy assessment framework via a **Biologically-Informed “Hybrid” Membership Inference Attack** (MIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. We show that our hybrid attack leads, on average, to higher adversarial success in non-DP models, and similar scores to traditional MIA on DP-enhanced ones, thus confirming that DP can successfully be leveraged for safer genomic data generation.

## Key Features

- **Differential Privacy**: Implementation of differentially-private mechanisms (DP-SGD) to ensure strong privacy guarantees
- **Language Model-Based Generation**: Implementation and comparison of language models (GPT-2 and minGPT) to generate realistic synthetic genetic mutation profiles  
- **Privacy-Utility Trade-off Analysis**: Comprehensive evaluation of the balance between privacy protection and data utility
- **Genomic Data Processing**: Tools for preprocessing and analyzing genetic mutation data
- **Evaluation Metrics**: Implementation of metrics to assess both privacy preservation and synthetic data quality

## Research Objectives

* **Privacy Preservation**: Develop methods that provide formal privacy guarantees for genomic data
* **Data Utility**: Maintain the statistical and biological relevance of synthetic genetic profiles

## Methodology

The project employs differentially-private language models to learn patterns in genetic mutation data and generate synthetic profiles that:

- Preserve population-level statistics and genetic associations
- Protect individual-level information through formal privacy guarantees
- Maintain biological plausibility and relevance for downstream analysis
- Enable safe data sharing for genomic research

## Repository Structure

```
├── data/                   # Data files and processing scripts
|   ├── sources/            # Source Data files (1000 Genomes Project)
|   |   ├── json/           # json-formatted source data files
|   |   ├── vcfs/           # vcf-formatted source data files
|   |   └── ...             
|   ├── generated/          # Model-Generated Data files
|   |   ├── json/           # json-generated files
|   |   └── vcfs/           # vcf-formatted generated files
|   ├── data_utils.py       # Scripts for data cleaning, analysis and processing
|   └── dataset.py          # Scripts for dataset formatting for model training
├── models/                 # Language Models and Tokenizer implementations
|   ├── saved/              # Saved model files^
|   |   ├── GPT/            # Saved finetuned GPT-2 model files
|   |   ├── minGPT/         # Saved minGPT model files
|   |   └── tokenizers/     # Saved Custom Tokenizers files
|   ├── finetuning.py       # Scripts for GPT-2 finetuning
|   ├── MinGPT.py           # Scripts for minGPT training
|   └── tokenizers.py       # Scripts for Custom Tokenizers
├── figures/                # Evaluation results figures
├── attacks.py              # Scripts for privacy attacks
├── evaluation.py           # Scripts for Utility and Rpivacy evaluation pipelines
├── metrics.py              # Scripts for utility metrics assessment
└── .ipynb                  # Research experiments files
```

> ^[!IMPORTANT]  
> Due to the large file sizes, we decided to not upload the trained models weights. Please [reach out](belfiore.asia.02@gmail.com) and request a copy of the weight, specifying your desired trained model.


## Evaluation

### Privacy Metrics
- **Formal Privacy Guarantees**: Verification of differential privacy bounds during training
- **Membership Inference Attacks**: Resistance to privacy attacks
- **Sample Distances**: Similarity and sitances between synthetic and real samples

### Utility Metrics  
- **Mutation Validity**: Format validity of the generated samples
- **Mutation Quality**: Biological plausibility of the generated samples
- **VCF Statistics**: Preservation of genetic associations
- **Variant Chromosomal Distribution**: Preservation of positional distribution of mutations within the selected chromosome (22)

## Results

- **Model Suitability**: GPT-like transformer-based models, both small and large-scale, are more than valid generators of synthetic genomic samples, both if trained from scratch and if finetuned. Smaller models tend to offer more inter-sample diversity within generated multi-sample cohorts, but at the cost of positional verisimilitude. Larger models, instead, are much more deterministic generators, thus generating cohorts of extremely similar samples.
- **Innate Privacy**: smaller transformers provide better innate privacy, with low memorization tendencies and good robustness against Membership Inference Attacks. Larger models, on the other hand, are more vulnerable to MIA attacks and tend to show much higher levels of memorization.
- **Differential Privacy**: implementing DP into the model training pipeline increases the privacy robustness of the model to inference attacks, and surprisingly, produces a regularization effect that enhances the model’s utility.
- **Hybrid Membership Inference**: we introduce a hybrid MIA attack which combines model metrics (like perplexity, loss and confidence) with genomic-specific features (like genotype frequencies, mutation rate and mutation frequencies by variant type) extracted from the model-generated sample to infer membership. This attack shows on average higher success rates compared to model-only based MIA, thus representing a promising approach for genomics-specific privacy assessments of generative models.

## Installation Notes

Full requirements can be found [here](requirements.txt) and [here](requirements_paperspace.txt) (for dp-transformers).

- Python 3.9+
- PyTorch
- Opacus
- dp-transformers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

> [!IMPORTANT]  
> Due to various package conflicts, we created multiple virtual enviroments with different package combinations and python versions. Please note that not all notebooks have been run in the same environment and, thus, may throw package errors when run together.

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Author Details

- **Author**: Asia Belfiore
- **Email**: belfiore.asia.02@gmail.com, 
             asia.belfiore24@imperial.ac.uk
- **Institution**: Imperial College London

## Acknowledgments

- Dissertation supervisors: Dr. *Jonathan Passerat-Palmbach*, Dr. *Dmitrii Usynin*
- Reources: Imperial College London Computing Department, DigitalOcean (formerly *Paperspace*) 
- Open-source genomic data provider: 1000 Genomes Project
<!-- - Privacy-preserving machine learning community: *Opacus* and *dp-transformers* developers -->