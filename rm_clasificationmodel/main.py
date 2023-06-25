
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
import pandas as pd


def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")

    df=DataFrame(es)

    df_eng=df.create()

    print(df_eng.head())

    # df_eng["category"].astype("category")

    # df =df_eng["category"].isnull()

    # bool_name_series = pd.notnull(df_eng["name"])
    # bool_cat_series = pd.isnull(df_eng["name"])
    # print(df_eng[bool_cat_series])
    




if __name__=="__main__":
    main()