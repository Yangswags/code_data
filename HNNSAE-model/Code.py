#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('Churn_Modelling.csv')

print ('This dataset contains {} rows and {} columns.'.format(df.shape[0], df.shape[1]))
df.head()


# In[2]:


df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)
df.columns


# In[3]:


df.head()


# In[4]:


df['Geography'].unique()


# In[5]:


df['Gender'].unique()


# In[6]:



df['Geography'] = df['Geography'].map({'France':1,'Spain':2,'Germany':3})
df['Gender'] = df['Gender'].map({'Female':0,'Male':1})


# In[7]:


df


# In[8]:


import toad
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor) 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# In[9]:


dev, off = train_test_split(df, test_size=0.2, random_state=328)
dep = 'Exited'
var_names = [i for i in list(df.columns) if i not in 'Exited']

X1 = dev[var_names].to_numpy()
y1 = dev[dep].to_numpy()

smo = SMOTE(random_state=42)
X_smo_1, y_smo_1 = smo.fit_resample(X1, y1)

X2 = off[var_names].to_numpy()
y2 = off[dep].to_numpy()

X_smo_2, y_smo_2 = smo.fit_resample(X2, y2)


# In[10]:


#Train data and Test data
x = torch.tensor(X_smo_1,dtype=torch.double)
y = torch.tensor(y_smo_1,dtype=torch.double)

val_x = torch.tensor(X_smo_2,dtype=torch.double)
val_y = torch.tensor(y_smo_2,dtype=torch.double)

del dev,off


# In[11]:


x_train = X_smo_1
y_train = y_smo_1
x_test = X_smo_2
y_test = y_smo_2


# In[12]:


#Gain centroid
n_bins=7
trans_var_names = var_names.copy()

combiner = toad.transform.Combiner()
combiner.fit(df[trans_var_names+[dep]],df[dep],method='quantile',n_bins=n_bins,exclude=[])
bins = combiner.export()
df_bin = combiner.transform(df[trans_var_names+[dep]])

for round_num,ft in enumerate(trans_var_names):
    bin_ary=np.array([])
    for i in range(n_bins):
        if np.isnan(df[df_bin[ft]==i][ft].mean())==False:
            avg = df[df_bin[ft]==i][ft].mean() 
        else:
            avg = -1
        bin_ary = np.append(bin_ary,avg)
    if round_num==0:
        bin_base=bin_ary.copy()
    else:
        bin_base = np.vstack((bin_base,bin_ary))
centroid=torch.tensor(bin_base,dtype=torch.double)
print(centroid.shape)


# In[13]:


class EntityEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(config.num_level, config.embedding_dim)
        self.centroid = config.centroid
        self.EPS=config.EPS
    def forward(self, x): 
        """
        x must be batch_size times 1
        """
        cent_hat = torch.tensor(self.centroid[0,:]).detach_().unsqueeze(1)
        x_hat = x[:,0].unsqueeze(1).unsqueeze(1)
        d = 1.0/((x_hat-cent_hat).abs()+self.EPS)
        w = F.softmax(d.squeeze(2), 1)
        v = torch.mm(w.type(torch.DoubleTensor), self.embedding.weight.type(torch.DoubleTensor))
        result = v.unsqueeze(1).type(torch.DoubleTensor)

        if x.size()[1]>1:
            for i in range(1,x.size()[1]):
                cent_hat = torch.tensor(self.centroid[i,:]).detach_().unsqueeze(1)
                x_hat = x[:,i].unsqueeze(1).unsqueeze(1)
                d = 1.0/((x_hat-cent_hat).abs()+self.EPS)
                w = F.softmax(d.squeeze(2), 1)
                v = torch.mm(w.type(torch.DoubleTensor), self.embedding.weight.type(torch.DoubleTensor)).type(torch.DoubleTensor)
                result = torch.cat((result, v.unsqueeze(1)), 1)
        return result


# In[14]:


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        #Caculate attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        #Restore Dimension
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


# In[15]:


import torch.nn.functional as F
class TransformerConfig:
    def __init__(self, 
                output_attentions = True,
                n_var=64,
                num_attention_heads = 24,
                hidden_size = 256,
                attention_probs_dropout_prob = 0.1,
                hidden_dropout_prob = 0.1,
                intermediate_size = 48,
                num_level=10,
                embedding_dim=64,
                EPS=1e-7,
                centroid={}):
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.n_var = n_var
        self.num_level=num_level
        self.embedding_dim = embedding_dim
        self.EPS=EPS
        self.centroid = centroid

class TransformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, 128)
        self.dense_2 = nn.Linear(128,64)
        self.dense_3 = nn.Linear(64,config.intermediate_size)
        self.dense_4 = nn.Linear(config.intermediate_size,config.intermediate_size)
        self.intermediate_act_fn = F.relu
        self.LB1 = nn.LayerNorm(128)
        self.LB2 = nn.LayerNorm(64)
        self.LB3 = nn.LayerNorm(config.intermediate_size)
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.LB1(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.LB2(hidden_states)
        hidden_states = self.dense_3(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.LB3(hidden_states)
        hidden_states = self.dense_4(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EntityEmbeddingLayer=EntityEmbeddingLayer(config)
        self.attention = SelfAttention(config)
        self.output = TransformerOutput(config)
        self.linear = nn.Linear(config.hidden_size*config.n_var, 1)
        
    def forward(self, hidden):
        result = self.EntityEmbeddingLayer(hidden.type(torch.DoubleTensor))
        self_attention_outputs = self.attention(result.type(torch.DoubleTensor))
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.output(attention_output.type(torch.DoubleTensor))
        outputs = (layer_output,)
        
        result = torch.sigmoid(self.linear(torch.flatten(layer_output.type(torch.DoubleTensor),start_dim=1)).squeeze(-1))
        
        return result


# In[16]:


transformer_config = TransformerConfig(output_attentions = True,
                n_var=len(trans_var_names),
                num_attention_heads = 2,
                hidden_size = 32,
                attention_probs_dropout_prob = 0.1,
                hidden_dropout_prob = 0.1,
                intermediate_size = 32,
                num_level=n_bins,
                embedding_dim=32,
                centroid=centroid)
transformer_layer = TransformerLayer(transformer_config)
transformer_layer


# In[ ]:



loss_fn = nn.BCELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.Adam(transformer_layer.parameters(), lr=learning_rate)

NUM_EPOCHS=1000
BATCH_SIZE=2048

for epoch in range(NUM_EPOCHS):
    
    for start in range(0,len(x),BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = x[start:end,:]
        batchY = y[start:end]
        y_pred = transformer_layer(batchX)
        loss = loss_fn(y_pred, batchY) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()          
    val_pred = transformer_layer(val_x).detach().numpy() 
    if epoch%10:
        print('No.{} epoch'.format(epoch), loss.item(),"test AUC",round(roc_auc_score(val_y.detach().numpy(), val_pred), 3))


# In[ ]:




