class gene_inference:
    
    def __init__(self,
                 output_path: str, 
                 model_path: str,
                 torchKG_path: str,
                 gene_symbol_list: list,
                 k: int ):     
        
        
        original_dir = os.getcwd()
        
        
        os.chdir('../graph_data/')
        genes_dir = os.getcwd()
        protein_coding_dir = os.path.join(genes_dir, 'protein-coding_gene.txt')
        self.df_protein_coding = pd.read_csv(protein_coding_dir, delimiter='\t', header=0, low_memory=False)
        filtered_df = self.df_protein_coding[self.df_protein_coding['symbol'].isin(gene_symbol_list)]
        self.gene_list = filtered_df['entrez_id'].tolist()
        
        
        os.chdir(original_dir)
        self.output_path = output_path
        
        try:
            os.mkdir(output_path)
        except OSError as error:
            raise error
        
        self.model = torch.load(model_path)
        torchKG = torch.load(torchKG_path)
        self.k = k
        
        self.d_ent = torchKG.ent2ix
    
        missing_genes = [x for x in self.gene_list if x not in list(torchKG.ent2ix.keys())]
        if not missing_genes:
            print("The graph contains all the genes of the list")
        else:
            error_message = f'The graph does not contain :{missing_genes}'
            raise ValueError(error_message)
            
            

    def find_genes(self):
        
        d_scores = {}
        
        gene_list_tensor = torch.tensor([self.d_ent[key] for key in self.gene_list], dtype=torch.int32)
        
        all_genes_encode = list(self.d_ent.values())
        
        for gene_all in all_genes_encode:

            #repeat the all_gene same length as the gene_list
            gene_all_tensor = torch.full((len(gene_list_tensor),),gene_all, dtype=torch.int32)

            #repeat the relation == 0 
            relation = torch.full((len(gene_list_tensor),),0, dtype=torch.int32)
            
            #comptute the summation of scores
            #i doesnt matter if we set head or tail for gene list thats why we use ComplEx
            
            d_scores[gene_all] = float(self.scoring_function(self.model,gene_all_tensor,relation,gene_list_tensor).sum(dim=0))


        
        d_scores_ID = {list(self.d_ent.keys())[i]: value for i, (_, value) in enumerate(d_scores.items())}   
            
        
        df_scores = pd.DataFrame({'encoding_ID': list(d_scores.keys()),'PubMed_ID': list(d_scores_ID.keys()), 'Σ_score': list(d_scores_ID.values())})
        filtered_df = self.df_protein_coding[self.df_protein_coding['symbol'].isin(df_scores['PubMed_ID'].tolist())]
        df_scores['symbol'] = filtered_df['symbol']
        df_scores = df_scores.sort_values(by='Σ_score', ascending=False)
        df_scores.to_json(f'{self.output_path}/scores_df.json')
        
        
        
        self.top_k_df = df_scores.head(self.k)
        filtered_df = self.df_protein_coding[self.df_protein_coding['symbol'].isin(self.top_k_df['PubMed_ID'].tolist())]
        print(filtered_df)
        self.top_k_df['symbol'] = filtered_df['symbol']
        self.top_k_df.to_json(f'{self.output_path}/top_K_df.json')
        
        
        self.pca_3Dplot()
    
    
    def pca_3Dplot(self):

        re_ent_emb = self.model.get_embeddings()[0]
        im_ent_emb = self.model.get_embeddings()[1]
        
        top_k_idx = list(self.top_k_df["encoding_ID"])
        gene_list_idx = [self.d_ent[key] for key in self.gene_list]
        
        features_idx = gene_list_idx + top_k_idx
        
        re_k_ent_emb = re_ent_emb[features_idx]
        im_k_ent_emb = im_ent_emb[features_idx]
        
        re_k_ent_emb_cpu = re_k_ent_emb.cpu()
        im_k_ent_emb_cpu = im_k_ent_emb.cpu()
        
        #The magnitude and phase of a complex number provide additional information about the complex number that is not captured by the real and imaginary parts alone.

        magnitudes = np.sqrt(re_k_ent_emb_cpu**2 + im_k_ent_emb_cpu**2)
        phases = np.arctan2(im_k_ent_emb_cpu, re_k_ent_emb_cpu)

        
        k_con_features = np.concatenate([re_k_ent_emb_cpu, im_k_ent_emb_cpu, magnitudes, phases], axis=1)

        scaler = StandardScaler()
        k_features = scaler.fit_transform(k_con_features)
        
        num_components = 3
        pca = PCA(n_components=num_components)
        
        reduced_features = pca.fit_transform(k_features)
        
        #cmap = plt.colormaps['viridis']
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        larger_size = 100

        for i in range(len(features_idx)):
            
            x, y, z = reduced_features[i]
            size = larger_size if i < len(gene_list_idx) else 35
            color = 'blue' if i < len(gene_list_idx) else 'red'
            #color = cmap(i / len(features_idx)) if i < len() else 'red'
            gene_symbol = list(self.df_protein_coding[self.df_protein_coding['entrez_id'].isin([list(self.d_ent.keys())[features_idx[i]]])]['symbol'])
            label = f"Gene {gene_symbol}" if i < len(gene_list_idx) else f"k-Gene {gene_symbol}"

            ax.scatter(x, y, z, color=color, s=size, label=label)

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('PCA 3D Visualization')

        plt.legend()

        output_file_path = f'{self.output_path}/pca_3d_plot.png'  
        plt.savefig(output_file_path, dpi=300)  # You can adjust the dpi value for image quality

        ret

    def scoring_function(self,model,h_idx,r_idx,t_idx):

        """Compute the real part of the Hermitian product
        :math:`\\Re(h^T \\cdot diag(r) \\cdot \\bar{t})` for each sample. 
        """
        re_ent_emb = model.get_embeddings()[0]
        im_ent_emb = model.get_embeddings()[1]
        re_rel_emb = model.get_embeddings()[2]
        im_rel_emb = model.get_embeddings()[3]

        re_h, im_h = re_ent_emb[h_idx], im_ent_emb[h_idx]
        re_t, im_t = re_ent_emb[t_idx], im_ent_emb[t_idx]
        re_r, im_r = re_rel_emb[r_idx], im_rel_emb[r_idx]

        return (re_h * (re_r * re_t + im_r * im_t) + im_h * (
                    re_r * im_t - im_r * re_t)).sum(dim=1)
        
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene Inference Script")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")
    parser.add_argument("--model_path", required=True, help="Path to the model file")
    parser.add_argument("--torchKG_path", required=True, help="Path to the torchKG file")
    parser.add_argument("--gene_list", nargs='+', required=True, help="List of genes")
    parser.add_argument("--k", type=int, required=True, help="Value of k")
    
    args = parser.parse_args()

    gene_inference_instance = gene_inference(args.output_path, args.model_path, args.torchKG_path, args.gene_list, args.k)
    gene_inference_instance.find_genes()