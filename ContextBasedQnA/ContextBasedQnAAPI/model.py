import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import torch.nn.functional as F

MAX_WORD_LENGTH = 40
D = 100

glove = GloVe('6B',100)


class CharEmbedding(nn.Module):

    def __init__(self,vocab_size,embedding_dim = 8, cnn_kernel_size = 5,word_embedding_size =100 ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=0,)
        self.cnn = nn.Conv1d(in_channels=embedding_dim,out_channels=word_embedding_size,kernel_size=cnn_kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size = MAX_WORD_LENGTH-cnn_kernel_size+1)
    
    def forward(self,x):
        batch_size = x.shape[0]
        x = x.view(-1,MAX_WORD_LENGTH)
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.cnn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.view(batch_size,-1,x.shape[1])
        return x

class WordEmbedding(nn.Module):
    def __init__(self):
        super().__init__()      
        self.embedding = nn.Embedding.from_pretrained(torch.cat((glove.vectors,torch.zeros(1,glove.dim)),dim=0))

    def forward(self,x):
        x=self.embedding(x)
        return x

class HighwayNetworkLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
               nn.Linear(glove.dim+D,glove.dim+D),
               nn.ReLU(),
               nn.Linear(glove.dim+D,glove.dim+D),
               nn.ReLU()
        )
        self.gate = nn.Sequential(
               nn.Linear(glove.dim+D,glove.dim+D),
               nn.Sigmoid()
        )
    def forward(self,x):
        x_transformed = self.transform(x)
        p = self.gate(x)
        return p*x_transformed + (1-p)*x

class ContextualEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(glove.dim+D,D,1,bidirectional = True,batch_first = True)

    def forward(self,x):
        x,_ = self.bilstm(x)
        return x

class AttentionFlowLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Linear(6*D,1)
    
    def forward(self,H,U):
        # H : contextual embedding of context
        # U : contextual embedding of query

        T = H.shape[1]
        J = U.shape[1]

        H_interleaved = torch.repeat_interleave(H,J,dim=1)
        U_repeated = U.repeat(1,T,1)

        assert(H_interleaved.shape==U_repeated.shape)

        HU = torch.cat((H_interleaved,U_repeated,H_interleaved*U_repeated),dim=-1)
        S = self.alpha(HU)
        S = S.view(-1,T,J)
        C2Q_att = F.softmax(S,dim = -1)
        U_tilde = torch.matmul(C2Q_att,U)
        Q2C_att = F.softmax(torch.max(S,dim=-1)[0],dim=-1)
        Q2C_att = Q2C_att.unsqueeze(1)
        H_tilde = torch.matmul(Q2C_att,H).repeat(1,T,1)
        G = torch.cat((H,U_tilde,H*U_tilde,H*H_tilde),dim=-1)
        return G,Q2C_att

class ModellingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(8*D,D,2,bidirectional = True,batch_first = True)
    def forward(self,G):
        M,_ = self.bilstm(G)
        return M

class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(10*D,1)
        self.dense2 = nn.Linear(10*D,1)
        self.bilstm = nn.LSTM(2*D,D,1,bidirectional = True,batch_first = True)
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self,G,M):
        GM = torch.cat((G,M),dim=-1)
        temp_GM = self.dense1(GM).squeeze(-1)
        start = self.softmax(temp_GM)
        M2,_ = self.bilstm(M)
        GM2= torch.cat((G,M2),dim=-1)
        temp_GM2 = self.dense2(GM2).squeeze(-1)
        end = self.softmax(temp_GM2)

        return start,end

class BiDAF(nn.Module):
    def __init__(self):
        super().__init__()  
        self.char_emb_layer=CharEmbedding(1311,embedding_dim=8)
        self.word_emb_layer=WordEmbedding()
        self.highway = HighwayNetworkLayer()
        self.cont_emb_layer=ContextualEmbedding()
        self.att_layer = AttentionFlowLayer()
        self.modelling_layer = ModellingLayer()
        self.output_layer = OutputLayer()
    
    def forward(self,context_char,context_word,query_char,query_word):
        context_char_emb = self.char_emb_layer(context_char)
        context_word_emb = self.word_emb_layer(context_word)
        final_context_word_embedding = torch.cat((context_char_emb,context_word_emb),dim = -1)
        final_context_word_embedding = self.highway(final_context_word_embedding)
        context_cont_emb = self.cont_emb_layer(final_context_word_embedding)

        query_char_emb = self.char_emb_layer(query_char)
        query_word_emb = self.word_emb_layer(query_word)
        final_query_word_embedding = torch.cat((query_char_emb,query_word_emb),dim = -1)
        # final_query_word_embedding = self.highway(final_query_word_embedding)
        query_cont_emb = self.cont_emb_layer(final_query_word_embedding)

        g,q2c_att = self.att_layer(context_cont_emb,query_cont_emb)
        m = self.modelling_layer(g)
        o = self.output_layer(g,m)
        return o,q2c_att


def load_model():
    global bidaf
    model_path = 'ContextBasedQnAAPI/bidaf_model_3'
    model_state_dict_path = 'ContextBasedQnAAPI/bidaf_model_state_dict'
    print('start')
    bidaf = BiDAF()
    print('made model')
    bidaf = torch.load(model_path)
    # bidaf.load_state_dict(torch.load(model_state_dict_path))
    print('dommne')