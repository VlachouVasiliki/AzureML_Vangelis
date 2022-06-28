


from azureml.core import  Dataset
from azureml.core import Run
import math
import pandas as pd

run = Run.get_context(allow_offline=False)
ws = run.experiment.workspace
datastore = ws.get_default_datastore()
#ws = Workspace(subscription_id, resource_group, workspace_name)



ds = Dataset.get_by_name(workspace=ws, name='vBankingPB')

df = ds.to_pandas_dataframe()


df = df.drop(columns = ['Customer_ID','Year','Month','Remmitances_In_nonGR_1Y_MaxAmount','Remmitances_In_nonGR_1Y_Amount'])

df = df.drop(columns = ['education_Code','Link_contacts_Last_3m_neu','Eb_Logins_3M_Months_num','Eb_Logins_3M_num','Legal_Person'])

df = df.drop(columns = ['City','Age_Band','Occupation','Global_SubSubSegment'])


def binF(x):
    if x is False:
        z = 0
    elif math.isnan(x):
        z = 0
    else:
        z =1
    return z


df['vBankingFlag'] = df['vBankingFlag'].apply(binF)


# In[ ]:


#dummy variables
for col in df.columns:       
       if df[col].dtypes=='object':
            #df.drop(columns=col, inplace = True)
            df = pd.get_dummies(df, prefix=col + '_', columns=[col])


# In[ ]:


for col in df.columns:       
       if df[col].dtypes=='datetime64[ns]':
            df = df.drop(columns = col)


# In[ ]:


def replace_gt(x):
    x = x.replace('<=','le ')
    x = x.replace('>=','ge')
    x = x.replace('>','gt ')
    x = x.replace('<','lt ')
    return x


# In[ ]:


df.columns = [replace_gt(x) for x in df.columns]


prepared_vBankingPB = Dataset.Tabular.register_pandas_dataframe(df, datastore, "prepared_vBankingPB", show_progress=True)