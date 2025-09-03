ASCENDgpt: A Phenotype-Aware Transformer for Cardiovascular Risk PredictionThis repository contains the official implementation for ASCENDgpt, a transformer-based model designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs), as described in the preprint:ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health RecordsChris Sainsbury*, Andreas Karwath**Equal contribution.📖 AbstractWe present ASCENDgpt, a transformer-based model specifically designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs). Our approach introduces a novel phenotype-aware tokenization scheme that maps 47,155 raw ICD codes to 176 clinically meaningful phenotype tokens, achieving 99.6% consolidation of diagnosis codes while preserving semantic information. We pretrain ASCENDgpt on sequences from 19,402 unique individuals using a masked language modeling objective, then fine-tune it for time-to-event prediction of five cardiovascular outcomes. Our model achieves excellent discrimination on a held-out test set with an average C-index of 0.816. The phenotype-based approach enables clinically interpretable predictions while maintaining computational efficiency.✨ Key FeaturesPhenotype-Aware Tokenization: Maps a vast set of 47,155 raw ICD codes to just 176 clinically meaningful phenotypes, drastically reducing vocabulary size and improving interpretability.Computational Efficiency: Achieves a 77.9% reduction in total vocabulary size, leading to a smaller model (103.3M parameters) and faster training times.High Predictive Performance: Demonstrates strong discrimination across five key cardiovascular outcomes.Survival Analysis Framework: Fine-tuned for time-to-event prediction using a Cox partial likelihood loss to handle right-censored clinical data appropriately.Interpretable Embeddings: The model learns a semantically rich embedding space where clinical concepts cluster in a medically coherent way.🚀 PerformanceASCENDgpt was fine-tuned to predict five cardiovascular outcomes within a 1-year window, based on a 5-year lookback of patient history. The model demonstrates strong and consistent performance across all outcomes on the held-out test set.OutcomeC-indexBrier Score (1-year)Myocardial Infarction (MI)0.7920.223Stroke0.8240.199MACE0.8000.181Cardiovascular Death0.8420.207All-Cause Mortality0.8240.223Average0.816-⚙️ Model & Training PipelineThe ASCENDgpt pipeline involves several key stages:Phenotype Mapping: Raw EHR data with 47,155 unique ICD codes is processed through a clinical mapping to consolidate them into 176 phenotype tokens.Sequence Construction: Patient histories are converted into sequences of tokens, including phenotypes, temporal information, and demographics.Pretraining: The transformer model (103.3M parameters) is pretrained on patient sequences using a Masked Language Modeling (MLM) objective to learn underlying patterns in the clinical data.Fine-tuning: The pretrained model is fine-tuned on a downstream survival prediction task using a Cox partial likelihood loss to predict risk scores for the five cardiovascular outcomes.🛠️ Getting StartedPrerequisitesPython 3.8+PyTorchTransformersPandasInstallationClone the repository:git clone [https://github.com/csainsbury/ASCENDgpt.git](https://github.com/csainsbury/ASCENDgpt.git)
cd ASCENDgpt
Install the required packages:pip install -r requirements.txt
Usage(Detailed instructions on how to preprocess data, pretrain, and fine-tune the model will be provided here.)# Example of how to use the model for prediction
from ascendgpt import ASCENDgptModel

# Load the fine-tuned model
model = ASCENDgptModel.from_pretrained("./models/fine_tuned_cvd")

# Prepare patient data sequence
patient_sequence = [...] # Tokenized patient EHR sequence

# Get risk prediction
risk_scores = model.predict(patient_sequence)
print(risk_scores)
📄 CitationIf you use ASCENDgpt in your research, please cite our work:@misc{sainsbury2024ascendgpt,
      title={ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health Records}, 
      author={Chris Sainsbury and Andreas Karwath},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={}
}
📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.📧 ContactChris Sainsbury - chris.sainsbury@nhs.scotAndreas Karwath - a.karwath@bham.ac.ukProject Link: https://github.com/csainsbury/ASCENDgpt
