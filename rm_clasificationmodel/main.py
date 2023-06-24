
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame

def main():
    es =ElasticSearchDb("http://172.17.212.35:9200")

    df=DataFrame(es)

    df_eng=df.create()

    print(df_eng)




if __name__=="__main__":
    main()