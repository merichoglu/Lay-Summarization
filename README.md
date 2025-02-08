# LaySummarization

## This repository contains scripts and configuration files for fine-tuning language models and generating summaries for lay-level understanding of scientific literature.

### Contents

- .gitignore: Ignores unnecessary files like logs and checkpoints.
- LaySummarization.docx: Document providing an overview or guide for the project.
- extract_plos_titles.py: Script to extract article titles from PLOS data.
- filter_titles.py: Filters article titles based on specific criteria.
- finetune.py: Fine-tunes a pretrained language model on the training dataset.
- finetune_config.json: Configuration file with parameters for fine-tuning.
- generate_summaries.py: Generates summaries for test data using the fine-tuned model.
- pmids_to_mesh.py: Maps PMIDs to MeSH terms.
- prepare_dataset_with_pmids.py: Prepares datasets with PMIDs for training.
- remove_notfound.py: Removes entries with missing data or errors.
- title_to_pmids.py: Converts article titles to PMIDs for dataset enrichment.
