import argparse
import pandas as pd
import os
import torch
from torchkge.data_structures import KnowledgeGraph


class kge_dataloader:
    
    def __init__(self, triples_df_file: str, 
                 train_size: float, 
                 validation: bool = True, 
                 output_path: str ='kge_data'):
        
        
        df_triples = pd.read_json(triples_df_file)
        self.train_size = train_size
        self.validation = validation
        self.output_path = output_path

        try:
            os.mkdir(self.output_path)
        except OSError as error:
            raise error

    def torchkge_data(self):
        torchKG = KnowledgeGraph(df=self.df_triples)
        print(f'Knowledge graph contain {torchKG.n_ent} genes')
        
        torch.save(torchKG, f"{self.output_path}/torchKG.pkl")

        train_kg, val_kg, test_kg = torchKG.split_kg(share=self.train_size, validation=self.validation)

        print(f'train set size: {train_kg.n_facts}')
        print(f'test set size: {test_kg.n_facts}')
        print(f'validation set size: {val_kg.n_facts}')

        torch.save(train_kg, f"{self.output_path}/train_kg.pkl")
        torch.save(val_kg, f"{self.output_path}/val_kg.pkl")
        torch.save(test_kg, f"{self.output_path}/test_kg.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Embedding DataLoader")

    # Add arguments for the required parameters
    parser.add_argument("--triples_df_file", type=str, help="Path to the triples DataFrame file")
    parser.add_argument("--train_size", type=float, help="Percentage of data to be used for training")
    parser.add_argument("--validation", action="store_true", help="Include validation set")
    parser.add_argument("--output_path", type=str, default="kge_data", help="Output path for the data")

   
    args = parser.parse_args()

    data_loader = kge_dataloader(
        triples_df_file=args.triples_df_file,
        relation_type=args.relation_type,
        train_size=args.train_size,
        validation=args.validation,
        output_path=args.output_path
    )

    data_loader.torchkge_data()
