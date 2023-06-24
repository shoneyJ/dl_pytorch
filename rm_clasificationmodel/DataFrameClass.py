import pandas as pd
class DataFrame:
    def __init__(self,es):
        self.es = es
        
    
    def create(self):
        df_eng = pd.DataFrame(columns=['name','category'])
        resp=self.es.getFeatures()

        for hit in resp['hits']['hits']:
            list_row =[]
            for name in hit['_source']['nameSource']:
                if (name["language"]=="en"):                    
                    list_row.append((name["value"]))     

    
            for cats in hit['_source']['categoriesSource']:
                if (cats["language"]=="en"):
                    list_row.append((cats["label"]))
            

            df_eng.loc[len(df_eng)] = list_row

        return df_eng      



