import streamlit as st ##vis/interactive
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

import pandas as pd #DS
import numpy as np

from typing import List, Tuple ##Misc
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
from sklearn.preprocessing import MinMaxScaler

##PyTorch
import torch
from torch import nn
import torch.nn.functional as F

##Some style things first
st.set_page_config(layout="wide")
sns.set_style('darkgrid')

train_preds = np.array([])

class VAE(nn.Module):
    def __init__(self, latent_size: int = 3):
        super(VAE, self).__init__()
        
        ##Encode layers
        self.encoder1 = nn.Linear(784, 512)
        self.encoder2 = nn.Linear(512, 128)
        
        ##Dist layers
        self.mu_layer = nn.Linear(128, latent_size)
        self.sigma_layer = nn.Linear(128, latent_size)
        
        ##Decode layers
        self.decoder1 = nn.Linear(latent_size, 512)
        self.decoder2 = nn.Linear(512, 648)
        self.decoder3 = nn.Linear(648, 784)
    
    def encode(self, x):    
        x = x.view(-1, 784)
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        
        ##To latent space
        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        return mu, sigma
    
    def decode(self, x):
        x = F.relu(self.decoder1(x))
        x = F.relu(self.decoder2(x))
        x = F.relu(self.decoder3(x))
        return x
    
    def forward(self, x):
        ##Encoder
        mu, sigma = self.encode(x)
        sample = mu + (sigma*torch.randn_like(mu))
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1).sum() * .5
        
        ##Decoder
        x = self.decode(sample)
        return x.view(-1, 28, 28)
    
    def get_embedding(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        
        ##To latent space
        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        sample = mu + (sigma*torch.randn_like(mu))
        return sample
    
    def generate_new_images(self, mu, sigma, count: int = 5):
        sample = mu.repeat(count, 1) + (sigma.repeat(count, 1) * torch.randn_like(mu.repeat(count, 1)))
        x = self.decode(sample)
        return x.view(-1, 28, 28)

##Load dataset. Cache this so it isn't reloaded everytime
@st.cache
def load_data() -> pd.DataFrame:
    data = pd.read_csv("./mnist_test.csv")
    X = MinMaxScaler().fit_transform(data.iloc[:, 1:].to_numpy())
    y = data.iloc[:, 0].to_numpy()
    return X.reshape(-1, 28, 28), y

@st.cache
def load_model():
    vae = VAE(2)
    vae.load_state_dict(torch.load("vae_model_lg1.pyt"))
    vae.eval()
    return vae

@st.cache
def get_embeds():
    embeddings = pyt_model.get_embedding(torch.tensor(X).float().view(-1,28,28)).detach()
    plot_data = pd.DataFrame({'Dim1': embeddings[:,0], 'Dim2': embeddings[:,1], 'Digit': y})
    return plot_data

def reset_session():
    st.session_state['model_history'] = []
    print('Cache cleared')

if 'model_history' not in st.session_state:
    st.session_state['model_history'] = []

##App Code
X, y = load_data()
pyt_model = load_model()
st.markdown("# Predicting and generating handwritten digits")

model_param_options = {
    'SVM': {
        'C': {'label':"C", 'min_value':1e-2, 'max_value':1e1, 'key':'svm', 'value':1., 'step':.1}
    },
    'Logistic Regression': {
        'penalty': {'label':'Penalty', 'options':['l2', 'l1']},
        'C': {'label':"C", 'min_value':1e-2, 'max_value':1e1, 'key':'lr', 'value':1., 'step':.1}
    }
}
model_objs = {
    'SVM': SVC,
    'Logistic Regression': LogisticRegression
}

##SIDEBAR CODE
with st.sidebar:
    st.markdown("## Model Configuration:")
    model_type = st.radio(
        label='Model type',
        options=['SVM', 'Logistic Regression']
    )
    selected_model_params = {}

with st.sidebar.expander("Model parameters", expanded=True):
    for param in model_param_options[model_type]:
        st_obj_args = model_param_options[model_type][param]
        if 'min_value' in st_obj_args:
            selected_model_params[param] = st.slider(**st_obj_args)
        elif 'options' in st_obj_args:
            selected_model_params[param] = st.radio(**st_obj_args)
        else:
            pass
 
with st.sidebar.expander('Data options', expanded=True):
    train_size = st.slider('Train size', .2, .9, value=.8)
    image_size = st.slider('Image size', 14, 28, value=14, step=14)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    ##Resize image
    input_size = 28
    output_size = image_size
    bin_size = input_size // output_size
    X_train = X_train.reshape((-1, output_size, bin_size, output_size, bin_size)).max(4).max(2).reshape(-1, output_size**2)
    X_test = X_test.reshape((-1, output_size, bin_size, output_size, bin_size)).max(4).max(2).reshape(-1, output_size**2)

with st.sidebar:
    if st.button('Train model'):
        ##Train model and get predictions
        model = model_objs[model_type](**selected_model_params).fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        st.session_state['model_history'].append(
            {
                "Model type": model_type,
                'Model params': selected_model_params,
                'Train accuracy': accuracy_score(y_train, train_preds),
                'Test accuracy':  accuracy_score(y_test, test_preds)
            }
        )

##MAIN TAB CODE
tab1, tab2 = st.tabs(["Predict (boring)", "Generate (cool)"])

with tab1:
    if train_preds.shape[0] != 0:
        col1, col2, col3 = st.columns(3)

        labels = list(range(0,10))
        train_cm = confusion_matrix(y_train, train_preds, labels=labels)
        test_cm = confusion_matrix(y_test, test_preds, labels=labels)
        
        train_text = [[str(y) for y in x] for x in train_cm]
        test_text = [[str(y) for y in x] for x in test_cm]

        with col1:
            st.markdown("## Train set")
            st.text("Accuracy = %f" % accuracy_score(y_train, train_preds))
            fig = ff.create_annotated_heatmap(
                train_cm, 
                x=labels, 
                y=labels, 
                annotation_text=train_text, 
                colorscale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("## Test set")
            st.text("Accuracy = %f" % accuracy_score(y_test, test_preds))
            fig = ff.create_annotated_heatmap(
                test_cm, 
                x=labels, 
                y=labels, 
                annotation_text=test_text, 
                colorscale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("## Leaderboard")
            st.button("Reset history", on_click=reset_session)
            st.dataframe(pd.DataFrame.from_records(st.session_state.model_history))
            
with tab2:
    plot_data = get_embeds()
    embed_chart = alt.Chart(plot_data).mark_point(
        opacity=.4,
        size=1,
    ).encode(
        x=alt.X('Dim1'),
        y=alt.Y('Dim2'),
        color=alt.Color('Digit:O', scale=alt.Scale(scheme='category10')),
    ).properties(height=500)
    st.altair_chart(embed_chart, use_container_width=False)


    dim1_start = st.number_input("Dim 1 start", value=-2.)
    dim1_end = st.number_input("Dim 1 end", value=1.5)
    dim2_start = st.number_input("Dim 2 start", value=-.5)
    dim2_end = st.number_input("Dim 2 end", value=-1.5)
    dim1 = (dim1_start, dim1_end)
    dim2 = (dim2_start, dim2_end)
    n = st.slider("Interpolation #", min_value=3, max_value=10, value=5)

    if st.button("Generate images"):
        fig, ax = plt.subplots(1,n, figsize=(3*n,3))
        for idx, val in enumerate(zip(np.linspace(*dim1, n), np.linspace(*dim2, n))):
                ax[idx].imshow(pyt_model.decode(torch.tensor(val).float()).detach().reshape(28,28), cmap='Greys')
                ax[idx].grid(False)
                ax[idx].set_xticks([])
                ax[idx].set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)