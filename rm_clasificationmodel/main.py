
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize


def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")

    df=DataFrame(es)

    [df_en, df_de]=df.create()

    vectorizer = Vectorize(1,1)

    vectorizer.fit(doc=df_en["name"])

    vectorizer.transform(df_en["name"])
   

    # df_eng["category"].astype("category")

    # df =df_eng["category"].isnull()

    # bool_name_series = pd.notnull(df_eng["name"])
    # bool_cat_series = pd.isnull(df_eng["name"])
    # print(df_eng[bool_cat_series])

    train = Train(vectorizer,df_en)

    train.run(10000)


    




if __name__=="__main__":
    main()