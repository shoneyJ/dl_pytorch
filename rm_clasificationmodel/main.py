
import torch
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize
from NN import RNN
from evaluate import Evaluate
from predict import *
import pandas as pd



def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")
   
    df=DataFrame(es)
    # dfProductTaxonomyEn=df.getProductTaxonomy(0,10000)    
    df_en = df.getNormal()
    # train = Train(df_en)
    # train.run(300000,30000,3000)
    predict = Predict()

    predict.confusionMatix(df_en)
    # # df_prediction=pd.DataFrame(columns=["id",'name','category','predicted','value'])
    # # list_row_en = dict (id=None,name=None,category=None,predicted=None,value=None)
    # list =[]
    # for index, row in dfProductTaxonomyEn.iterrows():
    #     predicted =predict.start(row["name"].lower())

    #     if(predicted[0][1]!=row["category"].lower()):
    #         list_row = dict (id=None,name=None,category=None,predicted=None,value=None)
    #         list_row["id"] =row["id"]
    #         list_row["name"] =row["name"]
    #         list_row["category"] =row["category"]
    #         list_row["predicted"] =predicted[0][1]
    #         list_row["value"] =predicted[0][0]

    #         list.append(list_row)

            

    #         # new_row = pd.Series(list_row)
    #         # df_prediction=pd.concat([df_prediction, new_row.to_frame().T], ignore_index=True)
    # es.createIndexerPrediction(list)


if __name__=="__main__":
    main()