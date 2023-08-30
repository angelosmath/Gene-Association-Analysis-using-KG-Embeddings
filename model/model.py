import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import torch
import torch.cuda as cuda
from torch.optim import Adam
from torchkge.models.bilinear import ComplExModel
from torchkge.sampling import UniformNegativeSampler
from torchkge.utils import MarginLoss, DataLoader


from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.evaluation import TripletClassificationEvaluator



class model:
    
    def __init__(self,input_path: str,
                 output_path: str, 
                 numEpochs: int,
                 embeddingDimension: int,
                 b_size: int,
                 n_neg: int,
                 learning_rate: float,
                 margin,
                 L2_term: float, 
                 evaluation: str):
        
        self.input_path = input_path
        self.output_path = output_path
        
        try:
            os.mkdir(output_path)
        except OSError as error:
            raise error
            
            
        if evaluation == 'best':
            self.eval_model = True
        elif evaluation == 'trained':
            self.eval_model = False
        else:
            raise ValueError("Invalid evaluation option. Use 'best' or 'trained'.")

        
        self.numEpochs = numEpochs
        self.b_size = b_size
        self.n_neg = n_neg
        self.learning_rate = learning_rate
        self.margin = margin
        self.L2_term = L2_term
        self.embeddingDimension = embeddingDimension
        
        self.train_kg = torch.load(f'{self.input_path}/train_kg.pkl')
        self.test_kg = torch.load(f'{self.input_path}/test_kg.pkl')
        self.val_kg = torch.load(f'{self.input_path}/val_kg.pkl')
        
    def model_initialization(self):
        
        self.complexModel = ComplExModel(emb_dim = self.embeddingDimension,
                            n_entities = self.train_kg.n_ent,
                            n_relations = self.train_kg.n_rel)
        

        total_params = sum(p.numel() for p in self.complexModel.parameters())
        print(f"Total Parameters: {total_params}")
        
        
    def train(self):
        
        # loss function
        criterion = MarginLoss(self.margin)
        
        # optimizer 
        optimizer = Adam(self.complexModel.parameters(), 
                         lr = self.learning_rate, 
                         weight_decay = self.L2_term)
        
        # negative sampling strategy
        sampler = UniformNegativeSampler(self.train_kg)
        
        
        # dataloader
        dataloader = DataLoader(self.train_kg, 
                                batch_size = self.b_size, 
                                use_cuda="all")
        
        
        if cuda.is_available():
            cuda.empty_cache()
            self.complexModel.cuda()
            criterion.cuda()
        
        #=========================================================
        #================= Training Loop =========================
        #=========================================================
        
        iterator = tqdm(range(self.numEpochs), unit='epoch')
        
        
        self.losses = []
        self.flt_mrrs = []
        flt_mrrs = 0
        best_mrr = 0.0
        
        
        for epoch in iterator:

            runningLoss = 0.0

            for batch in dataloader:

                head, tail, relation = batch[0], batch[1], batch[2]
                numHead, numTail = sampler.corrupt_batch(head, tail, 
                                                         relation,
                                                         n_neg = self.n_neg)

                optimizer.zero_grad()

                #forward - backward - optimize 
                pos, neg = self.complexModel(head, tail, relation, numHead, numTail)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()

            # Create an evaluator
            evaluator = LinkPredictionEvaluator(model = self.complexModel, 
                                                knowledge_graph = self.val_kg)

            # Perform evaluation without progress bar
            evaluator.evaluate(b_size = self.b_size,
                               verbose=False)

            mrr, flt_mrr = evaluator.mrr()
            
            # we save best model in training parametrs 
            if flt_mrr > best_mrr:
                self.best_model = self.complexModel #self.complexModel.state_dict()
            
            # early stopping 
            if flt_mrr > 0.9:
                self.best_model = self.complexModel #self.complexModel.state_dict()
                break

            
            self.losses.append(runningLoss/len(dataloader))
            self.flt_mrrs.append(flt_mrr)
            
            
            iterator.set_description('Epoch %d, loss %.5f , filtered MRR %.3f' % (epoch, runningLoss/len(dataloader),flt_mrr))
            #iterator.set_description('Epoch %d, loss %.5f' % (epoch, runningLoss/len(dataloader)))
        print(f'best filtered MRR {best_mrr}')
        
        #=========================================================
        #=========================================================

        
    def export_model(self):
        
        d = {'learning_rate': self.learning_rate,
             'batch_size': self.b_size,
             'negatives_sampels': self.n_neg,
             'regularization_term': self.L2_term}
        
        # Open the file in binary write mode
        pickle_file = open(f'{self.output_path}/dictionary_parametrs.pkl', "wb")

        # Save the dictionary to a Pickle file
        pickle.dump(d, pickle_file)

        # Close the file explicitly
        pickle_file.close()
        
        #saving best model 
        torch.save(self.best_model, f'{self.output_path}/best_model.pth')
        
        #saving trained model 
        torch.save(self.complexModel, f'{self.output_path}/model.pth')
        
    def evaluation(self):
        
        if self.eval_model:
            evl_model = self.best_model
        else:
            evl_model = self.complexModel
            
        evl_model.to('cpu')

        plot_path = f'{self.output_path}/plots'
        
        try:
            os.mkdir(f'{self.output_path}/plots')
        except OSError as error:
            raise error
        
        plt.plot(self.losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training')
        plt.savefig(f'{plot_path}/losses.png')
        
        
        
        plt.plot(self.flt_mrrs)
        plt.xlabel('Epochs')
        plt.ylabel('flt. MRR')
        plt.title('flt. MRR during training')
        plt.savefig(f'{plot_path}/flt_mrr.png')
        
        # Create an evaluator
        evaluator = LinkPredictionEvaluator(model=evl_model, 
                                            knowledge_graph=self.test_kg)

        # Perform evaluation without progress bar
        evaluator.evaluate(b_size = self.b_size, 
                           verbose=False)
        
        
        hit_10, flt_hit_10 = evaluator.hit_at_k()
        mrr, flt_mrr = evaluator.mrr()
        
        # Create bar plot for all metrics
        metrics = ['Hit@10', 'Filtered Hit@10', 'MRR', 'Filtered MRR']
        values = [hit_10, flt_hit_10, mrr, flt_mrr]

        fig, ax = plt.subplots()
        ax.bar(metrics, values)
        ax.set_ylabel('')
        ax.set_title('Link Prediction Evaluator Results')
        plt.savefig(f'{plot_path}/LinkPredictionEvaluatorResults.png')
        
        print('Link Prediction Evaluator Results')
        evaluator.print_results()
        
        # Triplet classification evaluation on test set by learning thresholds on validation set
        evaluator = TripletClassificationEvaluator(model=evl_model, 
                                                   kg_val=self.val_kg, 
                                                   kg_test=self.test_kg)
        
        
        print('Triplet Classification Evaluator Result')
        evaluator.evaluate(b_size=self.b_size)
        print('Accuracy on test set: {}'.format(evaluator.accuracy(b_size=100)))
        
        
        

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument("--input_path", type=str, required=True, help="Input path containing train, test, and val KG pickles.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to save model and evaluation results.")
    parser.add_argument("--numEpochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--embeddingDimension", type=int, required=True, help="Dimension of entity and relation embeddings.")
    parser.add_argument("--b_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--n_neg", type=int, required=True, help="Number of negative samples per positive sample during training.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the optimizer.")
    parser.add_argument("--margin", type=float, required=True, help="Margin for the MarginLoss.")
    parser.add_argument("--L2_term", type=float, required=True, help="L2 regularization term for the optimizer.")
    parser.add_argument("--evaluation", choices=["best", "trained"], default="trained", help="Evaluation mode ('best' or 'trained').")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Instantiate the model
    model_instance = model(args.input_path, args.output_path, args.numEpochs, args.embeddingDimension, args.b_size,
                           args.n_neg, args.learning_rate, args.margin, args.L2_term, args.evaluation)

    # Model initialization
    model_instance.model_initialization()

    # Train the model
    model_instance.train()

    # Export the model
    model_instance.export_model()

    # Perform evaluation
    model_instance.evaluation()
