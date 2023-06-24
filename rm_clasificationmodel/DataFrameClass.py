import pandas as pd
class DataFrame:
    def __init__(self,es):
        self.es = es
        
    
    def create(self):
        df_eng = pd.DataFrame(columns=['name','category'])
        resp=self.es.getFeatures()

        for hit in resp['hits']['hits']:
            list_row = dict (name=None,category=None)
            for name in hit['_source']['nameSource']:
                if (name["language"]=="en"):
                    list_row["name"]=name["value"]

    
            for cats in hit['_source']['categoriesSource']:
                if (cats["language"]=="en"):
                    list_row["category"]=cats["label"]
            

            if(list_row):
                new_row = pd.Series(list_row)
                df_eng=pd.concat([df_eng, new_row.to_frame().T], ignore_index=True)
            # df_eng.loc[len(df_eng)] = list_row

        return df_eng      



