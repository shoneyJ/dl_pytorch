
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize
from NN import RNN


def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")

    df=DataFrame(es)

    [df_en, df_de]=df.create()

    vectorizer = Vectorize(1,1)
    # vectorizer.setTfidFitTransform(df_en["name"])

    # x =vectorizer.getTfidFitTransform()

    # print(x.shape)

    vectorizer.fit(doc=df_en["name"])

    vectorizer.transform(df_en["name"]) 

    # df =df_eng["category"].isnull()

    # bool_name_series = pd.notnull(df_eng["name"])
    # bool_cat_series = pd.isnull(df_eng["name"])
    # print(df_eng[bool_cat_series])
    inputSize=vectorizer.getTransformedVectorSize()
    n_hidden = 256*6
    df_category = df_en.groupby('category')        
    all_category = list(df_category.groups.keys())        
    n_category = len(all_category)
    rnn=RNN(inputSize, n_hidden, n_category)

    train = Train(vectorizer,df_en,all_category,rnn)

    train.run(200000)


    




if __name__=="__main__":
    main()