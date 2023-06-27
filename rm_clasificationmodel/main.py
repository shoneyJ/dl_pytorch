
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize


def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")

    df=DataFrame(es)

    [df_en, df_de]=df.create()

    cv = Vectorize(1,1)

    cv.ngramFit(doc=df_en["name"])

    transformed_vector_name=cv.transform(df_en["name"])
   

    # df_eng["category"].astype("category")

    # df =df_eng["category"].isnull()

    # bool_name_series = pd.notnull(df_eng["name"])
    # bool_cat_series = pd.isnull(df_eng["name"])
    # print(df_eng[bool_cat_series])

    train = Train(cv,df_en)

    for i in range(10):
        category, name, category_tensor, name_tensor = train.randomTrainingExample()
    
        print('category =', category, ', name =', name)
    




if __name__=="__main__":
    main()