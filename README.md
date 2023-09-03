# Gene Association Analysis using Knowledge Graph Embeddings (under construction)

![Project Logo](https://snap-stanford.github.io/cs224w-notes/assets/img/node_embeddings.png?style=centerme)

## Overview

This repository contains the implementation and code developed for my master's thesis project. 
The project focuses on harnessing text mining techniques to construct a biomedical knowledge graph from curated data of research articles.
The graph is then analyzed using knowledge graph embedding techniques to uncover relationships between genes.

 You can find the BioCXML files provided by PubMed 
[here](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/PubTatorCentral_BioCXML/). 

All pieces of work conducted at the Computational BioMedicine Laboratory (CBML) in Institute of Computer Science (ICS) of Foundation for Research and Technology (FORTH).

## Installation

To clone this repository, use the following command:

```shell
git clone https://github.com/angelosmath/Gene-Association-Analysis-using-Knowledge-Graph-Embeddings
```

To install the dependecies, use the following command:

```shell
pip install -r requirements.txt
```

## Project Description


### `bioc_parser/` - BioC XML Parsing and Data Extraction

The `bioc_parser/` directory contains the code for parsing BioC XML files and extracting data for further analysis.
It parses the files to extract genes mentioned in the scientific articles, organizes them into a structured pandas DataFrame that is saved into JSON format.

***Execute the following command to run the script:***

   ```shell
   python bioc_parser.py --directory BIOCXML_DIR  --output_filename FILE_NAME
   ```
- `BIOCXML_DIR`: with the path to the directory containing your BioC XML files
- `FILE_NAME` desired name for the output JSON file

<br>
<br>

### `graph_data/` - Graph Generation and Visualization

The `graph_data/` directory contains the code for generating graph data in triples format (from,to,relation) connecting a pair of genes. The data derived from scientific articles that extracted from the the BioCXML files. A threshold was introduced to filter those relations based the amount of times a pair of genes occured in the parsed scientiffic artciles. Further, it removes not involved in the [protein coding process](https://www.genenames.org/download/statistics-and-files/) genes.


**Execute the following command to run the script:**

   ```shell
python graph_data.py --file_path FILE_NAME --counts_threshold threshold_integer --output_path OUTPUT_PATH
```

- `FILE_NAME`: Path to the input JSON file the `bioc_parser/` generate.
- `threshold_integer`: A value that determines the minimum number of counts a gene pair must have in order to be considered for further processing.
- `OUTPUT_PATH`: Path to the output folder for saving the graph files.


<br>
<br>


### `torch_dataloader/` - Knowledge Graph Embedding DataLoader

The `torch_dataloader/` directory contains the code for preparing knowledge graph data for training the knowledge graph embedding model. It convert the graph into a  [torchkge.data_structures.KnowledgeGraph](https://torchkge.readthedocs.io/en/latest/reference/data.html) object. Further more splits the graph data into train, test and validation sets.

**Execute the following command to run the script:**

   ```shell
   python kge_dataloader.py --triples_df_file TRIPLES_DF_FILE --train_size TRAIN_SIZE --validation --output_path OUTPUT_PATH
```

- `TRIPLES_DF_FILE`: Path to the input triples DataFrame file that `graph_data/` generate.
- `TRAIN_SIZE`:  Percentage of data to be used for training.
- `--validation`: Add this flag to include a validation set.
- `OUTPUT_PATH`: Path to the output folder for saving the data.



<br>
<br>

### `model/` - Knowledge Graph Embedding Model (ComplEx) Training and Evaluation

The `model/` directory contains the code that initializes, trains with early stopping (if filtered mean
reciprocal rank (MRR) exceed 0.9) and evaluates the performance on link prediction and triplet classification tasks of the models.The evaluation is performed either on the model that achieves the highest flt. MRR during training or the trained model at the user's option.

**Execute the following command to run the script:**

```shell
python model.py --input_path INPUT_PATH --output_path OUTPUT_PATH --numEpochs NUM_EPOCHS --embeddingDimension EMBEDDING_DIMENSION --b_size BATCH_SIZE --n_neg NUM_NEG_SAMPLES --learning_rate LEARNING_RATE --margin MARGIN --L2_term L2_TERM --evaluation EVALUATION_MODE
```

- `INPUT_PATH`: Path to the directory containing the output of `torch_dataloader/` generate.
- `OUTPUT_PATH`: Path where model parameters, trained models, and evaluation results will be saved.
- `NUM_EPOCHS`: Number of training epochs.
- `EMBEDDING_DIMENSION`: Dimension of entity and relation embeddings.
- `BATCH_SIZE`: Batch size used during training.
- `NUM_NEG_SAMPLES`: Number of negative samples per positive sample we want to generate during training.
- `LEARNING_RATE`: Learning rate for the optimizer.
- `MARGIN`: Margin for the MarginLoss function.
- `L2_TERM`: L2 regularization term for the optimizer.
- `EVALUATION_MODE`: Evaluation mode ('best' or 'trained').


<br>
<br>


### `gene_association/` - Gene Inference and Visualization

The `gene_association/` directory contains the code for gene association using knowledge graph embeddings. The script takes genes list, the Knowledge Graph Embeddings trained model and relates k genes from the graph to the input ones. The association of genes is performed by the sum of the scores of each gene in the graph with the provided by the user genes. It returns graph genes contains with the highest sums. All k-genes embeddings are visualized using PCA in a 3D space. 

**Execute the following command to run the script:**

```shell
python gene_association.py --output_path OUTPUT_PATH --model_path MODEL_PATH --torchKG_path TORCHKD_PATH --gene_list GENE_LIST --k K_VALUE
```

- `OUTPUT_PATH`: Path where results will be saved.
- `MODEL_PATH`:  Path to the trained model file that the `model/` provide.
- `TORCHKD_PATH`:  Path to the torchKG knowledge graph file that the `torch_dataloader/` provide.
- `GENE_LIST`: List of genes for inference. (eg. ARG2 RHOB)
- `K_VALUE`: the number of top associated genes to return. 

<br>
<br>

### `DisGenNet_evaluation/` - DisGenNet Database Evaluation

The `DisGenNet_evaluation/` directory contains the R script for querying the DisGeNET database to provide information about the relationships between genes and diseases. The script fetches disease-gene associations for the k associated genes that the `gene_association.py` script outputs and exports the results to a CSV file. 

Before you use the script you must change the `authorization.txt` with your [DisGenNet](https://www.disgenet.org/signup/#) account informations (email & password). 

**Execute the following command to run the script:**

```shell
Rscript DisGenNet_evaluation.R --genes top_K_df_path --output results.csv
```

- `top_K_df_path`: Path where the `gene_association.py` will saved the top_K_df.json.
- `results.csv`:  Path to save the results.



## References


- [PMC text mining subset in BioC: about three million full-text articles and growing](https://pubmed.ncbi.nlm.nih.gov/30715220/)
- [BioC: a minimalist approach to interoperability for biomedical text processing](https://pubmed.ncbi.nlm.nih.gov/24048470/)
- [Genenames.org: the HGNC resources in 2023 ](https://pubmed.ncbi.nlm.nih.gov/36243972/)
- [The DisGeNET knowledge platform for disease genomics](https://academic.oup.com/nar/article/48/D1/D845/5611674)
- [TorchKGE: Knowledge Graph Embedding in Python and PyTorch](https://arxiv.org/abs/2009.02963)
- [Complex Embeddings for Simple Link Prediction](https://arxiv.org/abs/1606.06357)
