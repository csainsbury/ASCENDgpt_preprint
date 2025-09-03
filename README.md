# ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)

## Overview

ASCENDgpt is a transformer-based model specifically designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs). Our approach introduces a novel phenotype-aware tokenization scheme that dramatically reduces computational complexity while maintaining strong predictive performance.

### Key Features

- **Phenotype-Aware Tokenization**: Maps 47,155 raw ICD codes to 176 clinically meaningful phenotype tokens (99.6% consolidation)
- **Efficient Architecture**: 103.3M parameters with 77.9% vocabulary reduction compared to raw ICD approaches
- **Strong Performance**: Achieves average C-index of 0.816 across five cardiovascular outcomes
- **Clinical Interpretability**: Operates on meaningful clinical concepts rather than granular diagnosis codes

## Paper

**Title**: ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health Records

**Authors**: Chris Sainsbury (NHS Greater Glasgow and Clyde), Andreas Karwath (University of Birmingham)

**Abstract**: We present ASCENDgpt, a transformer-based model specifically designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs). Our approach introduces a novel phenotype-aware tokenization scheme that maps 47,155 raw ICD codes to 176 clinically meaningful phenotype tokens, achieving 99.6% consolidation of diagnosis codes while preserving semantic information...

[Full paper PDF](./ASCENDgpt_preprint.pdf)

## Model Performance

| Outcome | C-index | Brier Score | Event Rate |
|---------|---------|-------------|------------|
| Myocardial Infarction | 0.792 | 0.223 | 3.8% |
| Stroke | 0.824 | 0.199 | 4.9% |
| MACE | 0.800 | 0.181 | 11.5% |
| Cardiovascular Death | 0.842 | 0.207 | 5.3% |
| All-cause Mortality | 0.824 | 0.223 | 7.8% |
| **Average** | **0.816** | - | - |

## Architecture

ASCENDgpt uses a transformer encoder architecture with:
- Vocabulary size: 10,442 tokens
- Hidden size: 768
- Number of layers: 12
- Attention heads: 12
- Max sequence length: 2,048
- Total parameters: 103.3M

## Installation

```bash
# Clone the repository
git clone https://github.com/csainsbury/ASCENDgpt.git
cd ASCENDgpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.13+
- Transformers 4.30+
- NumPy
- Pandas
- Scikit-learn
- lifelines (for survival analysis)

## Usage

### Data Preparation

```python
from ascendgpt.preprocessing import PhenotypeMapper

# Initialize phenotype mapper
mapper = PhenotypeMapper('mappings/icd_to_phenotype.json')

# Convert ICD codes to phenotypes
phenotypes = mapper.map_codes(icd_codes)
```

### Model Training

```python
from ascendgpt.model import ASCENDgpt
from ascendgpt.trainer import Trainer

# Initialize model
model = ASCENDgpt(
    vocab_size=10442,
    hidden_size=768,
    num_layers=12,
    num_heads=12
)

# Setup trainer
trainer = Trainer(
    model=model,
    train_data=train_loader,
    val_data=val_loader,
    learning_rate=5e-6
)

# Train
trainer.pretrain(epochs=50)
trainer.finetune(task='survival', outcomes=['MI', 'stroke'])
```

### Prediction

```python
# Load pretrained model
model = ASCENDgpt.from_pretrained('path/to/checkpoint')

# Make predictions
risk_scores = model.predict(patient_sequences)
```

## Dataset

The model was developed using the INSPECT dataset, containing:
- 19,402 unique patients
- Median follow-up: 6.8 years
- 47,155 unique ICD codes mapped to 176 phenotypes

For access to the INSPECT dataset, please refer to [Huang et al., 2023](https://arxiv.org/abs/2311.09164).

## Repository Structure

```
ASCENDgpt/
├── ascendgpt/
│   ├── model.py           # Model architecture
│   ├── preprocessing.py   # Data preprocessing and phenotype mapping
│   ├── tokenization.py    # Custom tokenizer
│   ├── trainer.py         # Training logic
│   └── utils.py          # Utility functions
├── mappings/
│   ├── icd_to_phenotype.json  # ICD to phenotype mappings
│   └── phenotype_categories.json  # Clinical categorization
├── configs/
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   └── demo.ipynb        # Demonstration notebook
├── tests/
│   └── test_model.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use ASCENDgpt in your research, please cite:

```bibtex
@article{sainsbury2025ascendgpt,
  title={ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health Records},
  author={Sainsbury, Chris and Karwath, Andreas},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- INSPECT dataset creators for providing the data foundation
- Life2Vec for inspiring the sequence modeling approach
- BEHRT and Hi-BEHRT for pioneering transformer applications to EHR data

## Contact

- Chris Sainsbury: chris.sainsbury@nhs.scot
- Andreas Karwath: a.karwath@bham.ac.uk

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Disclaimer

This model is for research purposes only and has not been validated for clinical use. Always consult healthcare professionals for medical decisions.
