
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize


def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")

    df=DataFrame(es)

    [df_en, df_de]=df.create()

    print(df_en["name"].unique())

    cv = Vectorize(1,1)

    cv.ngramFit(doc=df_en["name"])

    transformed_vector_name=cv.transform(df_en["name"])
    vocab = cv.getVocabulary()

    print (transformed_vector_name.shape)

    # df_eng["category"].astype("category")

    # df =df_eng["category"].isnull()

    # bool_name_series = pd.notnull(df_eng["name"])
    # bool_cat_series = pd.isnull(df_eng["name"])
    # print(df_eng[bool_cat_series])
    




if __name__=="__main__":
    main()