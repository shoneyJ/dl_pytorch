
import torch
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize
from NN import RNN
from evaluate import Evaluate



def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")
   
    df=DataFrame(es)
    dfProductTaxonomyEn=df.getProductTaxonomy(0,10000)

    
    # df_en = df.getNormal()

    # [df_en, df_de]=df.create()

    


    # rnn=torch.load('ngram-rnn-classification.pt')
   
    # train = Train(df_en)
    # train.run(20)
    # train.predict('vaico  centering bush propshaft',2)

    


if __name__=="__main__":
    main()