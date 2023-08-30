import bioc 
import lxml
from tqdm import tqdm
import os
import glob
import re
import pandas as pd
import argparse

class parse_bioc:

    def __init__(self,
                 directory:str, 
                 output_filename: str):
        
        self.xml_names = self.get_filenames(directory)
        self.output_filename = output_filename
    
    def parsing(self):
        
        self.papers = []
        counter = []
        
        for i in tqdm(range(len(self.xml_names)),ascii = True,desc = 'progress'):
            bioc_file = self.parse_bioc(self.xml_names[i])
            if bioc_file == None:
                f = re.search(r'/([^/]+\.BioC\.XML)$' , self.xml_names[i]).group(1)
                print(f'{f} could not be parsed')
                continue
            #self.documents[re.search(r'/([^/]+\.BioC\.XML)$', self.xml_names[i]).group(1)] , n_papers = (self.parse_collection(bioc_file))
            c , n_papers = self.parse_collection(bioc_file)
            self.papers += c
            counter.append(n_papers)
                
        #print(f'{len(self.collections)} XML files parsed from {len(self.xml_names)}')
        print(f'we found {len(self.papers)} papers')
        print(f'{round(sum(counter)/len(counter))} mean number of papers per collection')
    
    
    def export_pandas(self):
        
        #rows = []
        #for _, value in self.documents.items():
        #    for row in value:
        #        rows.append(row)  

        columns = ['paper_ID', 'Text', 'Genes']
        df = pd.DataFrame(self.papers, columns=columns)
        df_filtered = df[df['Genes'].apply(lambda x: bool(x))]  #remove papers that not contain genes
        print(f'{df_filtered.shape[0]} papers found that refer genes')
        df_filtered.to_json(f'{self.output_filename}.json', orient='records', lines=False)
    
    def get_filenames(self,xml_directory):

        filenames_wild = os.path.join(xml_directory, '*.XML')
        filenames = glob.glob(filenames_wild)

        print ('Found {} BioC XML files'.format(len(filenames)))
        
        return filenames


    def parse_bioc(self,filename):
        
        try:
            with open(filename,encoding='utf-8') as f: #,encoding='Latin1'
                text = f.read()
                try:
                    return bioc.loads(text)
                except lxml.etree.XMLSyntaxError as e:
                    return None
        except (UnicodeDecodeError,FileNotFoundError):
            return None 
            
    
    def parse_collection(self,collection):
        
        documents_infos = []
        c_paper = 0
        
        for document in collection.documents:
            
            c_paper +=1
            
            id_ = document.id
            genes = set()
            text = set()
            
            for passage in document.passages:
                
                text.add(passage.text)
                
                for annotation in passage.annotations:

                    if annotation.infons['type'] == 'Gene':
                        try:
                            if annotation.infons['identifier'] == None:
                                continue
                            else:
                                gene_ids = annotation.infons['identifier'].split(';')
                                valid_gene_ids = [int(x) for x in gene_ids if x is not None and x.isdigit()] #ncbi_gene_ids
                                genes |= set(valid_gene_ids) #ncbi genes id 
                        except KeyError as e:
                            continue

                            
            infos  = [id_,text,genes]
            documents_infos.append(infos)
        return documents_infos, c_paper
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse BioCXML files for genes')
    parser.add_argument('--directory', type=str, help='Path to the BioCXML files')
    parser.add_argument('--output_filename', type=str, default='./papers_df', help='output dataframe file name')

    args = parser.parse_args()


    parser = parse_bioc(directory = args.directory,
                        output_filename = args.output_filename)

    parser.parsing()
    
    parser.export_pandas()