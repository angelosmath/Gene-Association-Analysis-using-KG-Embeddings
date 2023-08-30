import pandas as pd
import itertools
import argparse
import os 
from tqdm import tqdm
import json
import requests

class GraphData:
    
    def __init__(self, 
                 file_path: str, 
                 counts_threshold: int, 
                 output_path: str = "./graph_data"):
        
        self.file_path = file_path 
        self.output_path = output_path
        self.counts_threshold = counts_threshold
        
        try:
            os.mkdir(self.output_path)
        except OSError as error:
            raise error

        self.feature_df = self.genes_df()
        
        self.get_edges()

        
    def genes_df(self):
        
        df = pd.read_json('/home/angelosmath/MSc/thesis_genes/bioc_august/test_df_200.json')
        
        # remove papers that refer only 1 gene
        mask_single_gene = df['Genes'].apply(lambda x: len(x) == 1)
        df_filtered = df[~mask_single_gene]

        print("found {} papers that contain pairs of genes".format(df_filtered.shape[0]))
        
        return df_filtered

    def get_edges(self):

        self.pair_counts = {}

        for _, row in tqdm(self.feature_df.iterrows()):
            
            feature = row['Genes']
            pairs = list(itertools.combinations(feature, 2))

            for pair in pairs:
                pair = tuple(sorted(pair))  # Sort to ensure consistency
                if pair in self.pair_counts:
                    self.pair_counts[pair] += 1
                else:
                    self.pair_counts[pair] = 1
        return self.pair_counts
        
    def export(self):
        
        pair_counts_str_keys = {str(pair): count for pair, count in self.pair_counts.items()}
        
        with open(f'{self.output_path}/pairs_dictionary.json', 'w') as json_file:
            json.dump(pair_counts_str_keys, json_file)

        df = pd.DataFrame(list(self.pair_counts.items()), columns=['pair', 'count'])
        df = df.drop_duplicates(subset='pair')
        df[['from', 'to']] = pd.DataFrame(df['pair'].tolist(), index=df.index)
        df.drop(columns=['pair'], inplace=True)
        df['rel'] = 'gene-->gene'
        df = df[['from', 'to', 'rel', 'count']]

        # Download protein-coding gene data
        protein_coding_url = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/locus_groups/protein-coding_gene.txt"
        protein_coding_file = "protein-coding_gene.txt"
        os.system(f"wget {protein_coding_url} -O {protein_coding_file}") 
        df_protein_coding = pd.read_csv('protein-coding_gene.txt', delimiter='\t', header=0, low_memory=False)

        
        # Filter out rows with 'from' and 'to' genes not in df_protein_coding
        valid_genes = set(df_protein_coding['entrez_id'])
        df = df[df['from'].isin(valid_genes)]
        df = df[df['to'].isin(valid_genes)]
                
        #drop the rows that count is lower or equal to the threshold 
        df = df[df['count'] > self.counts_threshold]
        
        #find the unique genes we have in the graph 
        unique_genes = set(df['from']).union(df['to'])
        num_unique_genes = len(unique_genes)
        
        df_graph = df.drop(columns=['count'])
        
        df_graph.to_json(f'{self.output_path}/graph_df.json', orient='records', lines=False) 
        
        df.to_json(f'{self.output_path}/graph_df_counts.json', orient='records', lines=False) 
        
        print(f'The threshold aplied and the genes checked if they involved in protein coding. We have {df.shape[0]} pairs.')
        
        print("Number of unique genes:", num_unique_genes)
        
        print('Done')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process graph data.')
    parser.add_argument('--file_path', type=str, help='Path to the JSON file')
    parser.add_argument('--counts_threshold', type=int, help='Counts threshold')
    parser.add_argument('--output_path', type=str, default='./graph_data', help='Output path')

    args = parser.parse_args()


    graph_data = GraphData(
        file_path=args.file_path,
        counts_threshold=args.counts_threshold,
        output_path=args.output_path
    )

    graph_data.export()