# Spam Detection Project

## Overview

This repository contains notebooks and datasets for a spam detection model, with a strong emphasis on text processing and analysis. The project includes three main Jupyter notebooks, each of which focuses on different stages of text data handling and analysis, ranging from pre-processing to advanced feature engineering and detailed linguistic analysis. The dataset used in this project is `spam_ham_dataset.csv`.

## Repository Structure

- `Pre-Processing_Summary.ipynb`: Notebook for cleaning and pre-processing the text data.
- `Feature_Engineering.ipynb`: Notebook for feature extraction from the processed text.
- `Step_Analysis.ipynb`: Notebook for linguistic analysis, including morphological, syntactic, and semantic analysis.
- `spam_ham_dataset.csv`: The dataset used in this project.

## Files

### 1. `Pre-Processing_Summary.ipynb`

This notebook focuses on preparing the raw text data for further analysis and model training. The key steps are:

- **Punctuation Removal**
- **URL Filtering**
- **Spelling Correction**
- **Lowercasing**
- **Tokenization**
- **Stopword Removal**
- **Stemming and Lemmatization**

### 2. `Feature_Engineering.ipynb`

This notebook handles feature extraction techniques to convert the pre-processed text into formats that machine learning models can understand:

- **Bag of Words (BoW)**: Term frequency matrix creation.
- **TF-IDF**: Term frequency-inverse document frequency transformation.
- **Word Embeddings**: Pre-trained embeddings such as Word2Vec.
- **N-grams**: Extracting word pairs and triplets to capture context.
- **Metadata Extraction**: Features like email length and presence of keywords.

### 3. `Step_Analysis.ipynb`

This notebook dives into various levels of linguistic analysis, covering:

#### a. **Lexical and Morphological Analysis**

- **Tokenization**: Breaking down the text into morphemes (stems, affixes).
- **Spacy Library**: Used to identify morphological features (e.g., singular/plural, tense).
  
  Example output of tokenized text:
  ```
  Token                Morphological Analysis
  This                 Number=Sing|PronType=Dem
  is                   Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
  an                   Definite=Ind|PronType=Art
  example              Number=Sing
  sentence             Number=Sing
  ```

#### b. **Morphosyntactic Analysis (POS Tagging)**

- **Spacy, NLTK, and TextBlob** are used to perform Part-of-Speech tagging.
- Identifies grammatical roles of words (e.g., noun, verb, adjective).


#### c. **Syntactic Analysis (Parsing)**

- **Spacy Dependency Parsing**: Visualizes and analyzes sentence structure using dependency trees.
- **Chunking**: Grammatical chunking with custom rules using NLTK's `RegexpParser`.

- **Tree Visualizations**: Displays sentence structure graphically using libraries such as `displaCy` (for dependency parsing) and NLTK's tree representations.

#### d. **Semantic and Pragmatic Analysis**

- **Word Sense Disambiguation (WSD)**: Analyzes word meanings in context using advanced NLP tools.
- **Pragmatic Analysis**: Considers word usage and context for nuanced meaning.

#### Example Visualizations

- **Dependency Parsing**: Graphically represents sentence structure using Spacy's `displacy` tool.
- **Syntactic Trees**: Syntax trees visualize the hierarchical relationships between parts of the sentence.

### 4. `spam_ham_dataset.csv`

This file contains the dataset used in this project. It includes thousands of email samples, labeled either "spam" or "ham". The main columns are:

- **label**: Class of the email (spam/ham).
- **text**: The raw text of the email.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/spam-detection.git
   cd spam-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open Jupyter notebooks in the following order:
   - First, run `Pre-Processing_Summary.ipynb` to clean and prepare the dataset.
   - Then, run `Feature_Engineering.ipynb` to extract features from the cleaned text.
   - Finally, run `Step_Analysis.ipynb` to perform detailed linguistic analysis.

## Dependencies

- Python 3.8+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- NLTK
- spaCy
- TextBlob
- svgling (for tree visualizations)

To install all required libraries, use:
```bash
pip install -r requirements.txt
```

## Conclusion

This project showcases a full spam detection pipeline, from data cleaning to feature engineering and linguistic analysis. The advanced linguistic analysis in the `Step_Analysis.ipynb` file can also be adapted for other natural language processing tasks, such as translation or text generation.
