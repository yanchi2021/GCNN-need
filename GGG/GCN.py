import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy.sparse as sp
from build_graph import preprocess_adj,normalize_adj,node_size,adj,train_X,test_X,train_Y,test_Y,X_all,Y_all,vocab_size
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser

# In[101]:
class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x):
        a=x._values().size()
        b=torch.rand(a)

        mask=((b+(self.kprob)).floor()).byte()
        x_i=x._indices()
        rc=x_i[:,mask]
        data=x._values()
        val=data[mask]*(1.0/self.kprob)
        return torch.sparse.FloatTensor(rc, val,torch.Size([x.shape[0],x.shape[1]]))
class GCN_Layer(nn.Module):
    def __init__(self,units,A_hat,X_size,args,use_bias=True):
        super(GCN_Layer, self).__init__()
        self.use_bias=use_bias
        self.units=units
        self.X_size=X_size
        A_hat_i=torch.LongTensor([A_hat.row,A_hat.col])
        A_hat_v=torch.FloatTensor(A_hat.data)
        self.A_hat=torch.sparse_coo_tensor(A_hat_i, A_hat_v, torch.Size([node_size,node_size]))
        self.weight=nn.parameter.Parameter(torch.FloatTensor(X_size, args.hidden_size_1))
        var = 2. / (self.weight.size(1) + self.weight.size(0))
        nn.init.xavier_uniform(self.weight,gain=1.0)

        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_1, args.num_classes))
        var2 = 2. / (self.weight2.size(1) + self.weight2.size(0))
        nn.init.xavier_uniform(self.weight2, gain=1.0)

        if use_bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_1))
            nn.init.zeros_(self.bias)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(args.num_classes))
            nn.init.zeros_(self.bias2)
        else:
            self.register_parameter("bias", None)

    def forward(self,X):

        X=X.tocoo()
        X_i = torch.LongTensor([X.row, X.col])
        X_v = torch.FloatTensor(X.data)
        X=torch.sparse_coo_tensor(X_i,X_v,torch.Size([X.shape[0],X.shape[1]]))
        # 先drop out
        Drop_sp=SparseDropout(0.5)
        X=Drop_sp(X)
        X=torch.spmm(X,self.weight)
        if self.use_bias:
            X=X+self.bias
        X=torch.spmm(self.A_hat,X)
        X=F.relu(X)
        Drop_1 = nn.Dropout(0.5)
        X=Drop_1(X)
        X=torch.spmm(X,self.weight2)
        if self.use_bias:
            X=X+self.bias2
        X = torch.spmm(self.A_hat, X)
        # Drop_2 = nn.Dropout(0.5)
        # X=Drop_2(X)
        return X

# In[102]:
def categorical_crossentropy(preds, labels):   # """    :param preds:模型对样本的输出数组    :param labels:样本的one-hot标签数组    :return:样本的平均交叉熵损失    """    
    # np.extract(condition, x)函数，根据某个条件从数组中抽取元素    # np.mean()函数默认求数组中所有元素均值    
    a=preds[vocab_size:vocab_size+len(train_Y)]
    b=labels[0:len(train_Y)]
    return np.mean(-np.log(np.extract(a, b)))
def compute_mask(idx,l):
    mask=np.zeros(l)
    mask[idx]=1
    return np.array(mask,dtype=np.bool)
# idx_train=np.arange(vocab_size,vocab_size+train_len)
# sample_mask=compute_mask(idx_train,node_size)
def evaluate(output,label):
    a=output.numpy().argmax(1)
    b=label
    num=np.sum(a==b)
    return num/len(a)
def evaluate2(output, label):
    a = output.argmax(1)
    b = label
    num = np.sum(a == b)
    return num / len(a)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--hidden_size_1", type=int, default=100, help="Size of first GCN hidden weights")#mr-200-60

    parser.add_argument("--num_classes", type=int, default=2, help="Number of prediction classes")        #R8-200-80
    # parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    parser.add_argument("--num_epochs", type=int, default=100, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    parser.add_argument("--use_pretrain", type=bool, default=True, help="Model ID")

    args = parser.parse_args()
    # save_as_pickle("args.pkl", args)
    num_max=max(train_Y)
    args.num_classes=num_max+1
    adj = preprocess_adj(adj)
    adj_hat = normalize_adj(adj)
    adj_hat_coo = adj_hat.tocoo()
    from sklearn.preprocessing import OneHotEncoder
    # to one-hot
    # enc = OneHotEncoder(sparse=False)

    y_=np.array(Y_all)
    # y_train=y_[vocab_size:vocab_sizelen(),:]
    if(args.use_pretrain):
        embed_word=np.load('embedding.npy')
        embed_word_shape=embed_word.shape#194903,768
        # embed_word=embed_word[:embed_word_shape[0]-1][:]
        doc_emed=np.zeros([node_size-vocab_size,embed_word_shape[1]])
        x_=np.vstack((embed_word,doc_emed))
        x_=sp.csr_matrix(x_)
        print('load embedding done')
    else:
        x_ = sp.eye(node_size, node_size)
    train_len = len(train_Y)
    selected=np.arange(vocab_size,vocab_size+train_len)
    labels_selected=y_[0:train_len]
    labels_not_selected=y_[train_len:]

    # f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args)
    net = GCN_Layer(x_.shape[1], adj_hat_coo,x_.shape[1], args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50, 500, 1000, 5000, 6000], gamma=0.77)
    # start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=True)
    # losses_per_epoch, evaluation_untrained = load_results(model_no=args.model_no)

    print("Starting training process...")
    net.train()
    evaluation_trained = []
    for e in range(0, args.num_epochs):
        optimizer.zero_grad()
        output = net(x_)
        a=output[selected]
        b=torch.tensor(labels_selected).long()
        loss = criterion(a,b)#-1
        # losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if e % 2 == 0:
            ### Evaluate other untrained nodes and check accuracy of labelling
            net.eval()
            with torch.no_grad():
                pred_labels = net(x_)

                trained_accuracy = evaluate(output[selected], labels_selected);
                untrained_accuracy = evaluate(pred_labels[vocab_size+train_len:], labels_not_selected)
                # save_np=pred_labels[vocab_size + train_len:]
                # save_np=save_np.numpy()
                # np.save(str(e)+'Rw22.npy',save_np)
                # label_saved=np.array(labels_not_selected)
                # np.save(str(e)+'Rl22.npy',label_saved)
                # plt.figure(figsize=(8, 4))
                # fea2 = TSNE(n_components=2).fit_transform(save_np)
                # embe_list2 = []
                # for i in range(save_np.shape[0]):
                #     embe_list2.append([])
                # for num, index in enumerate(labels_not_selected):
                #     embe_list2[index].append(fea2[num])
                # plt.subplot(122)
                # for i, w_e in enumerate(embe_list2):
                #     if len(w_e) > 0:
                #         np_w = np.array(w_e)
                #         a = np_w[:, 0]
                #         b = np_w[:, 1]
                #         plt.scatter(a, b, label=i)
                #     else:
                #         continue
                # # plt.legend()
                # plt.title('2nd layer')
                # # print(evaluate(w2, test_Y))
                # plt.savefig(str(e)+'doc_word.png')
            # evaluation_trained.append((e, trained_accuracy));
            # evaluation_untrained.append((e, untrained_accuracy))
            print("[Epoch %d]: loss: %.7f" % (e, loss.data))

            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))

            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))

            # print("Labels of trained nodes: \n", output[selected].max(1)[1])

            net.train()

        # scheduler.step()





# In[ ]:




