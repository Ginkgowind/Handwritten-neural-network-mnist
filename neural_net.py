import numpy as np
import math
import data_type as dt
import test_module as testm

class nn:
    def __init__(self,train_num,training_epochs,batch_size,lerning_rate,train_x,train_y,test_x,test_y):
        self.train_num = train_num
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = lerning_rate
        self.total_step = int(self.train_num / self.batch_size)
        self.train_x=train_x
        self.train_y=train_y
        self.test_x = test_x
        self.test_y = test_y

        self.dense = []
        self.d_dense = None
        self.ret = []
        self.relu = []

    def forward(self, xs,ys,relu,softmax_loss):
        # 前向传播
        dense_num=len(self.dense)
        w=self.dense    #dense没被改变
        for i in range(dense_num):
            w_num=len(w[i][0])
            # if i <= dense_num
            if i == 0:
                ret1 = np.dot(xs,w[i])
            else:
                ret1 = np.dot(ret3,w[i])

            if i < dense_num-1:
                arr_one = np.ones(self.batch_size)  #增广维数固定
                #增广1
                # print(ret1.shape,w_num,arr_one.shape)
                ret2 = np.insert(ret1, w_num, arr_one, axis=1) # (5,65)
                # ret3 = relu.forward(ret2)
                ret3 = self.relu[i].forward(ret2)
                self.ret[i]=ret2   # 每一次的ret,供反向传播使用
            else :  #i == dense_num - 1
                #计算完隐藏层，开始计算输出层
                # ret1就是预测分数矩阵
                thescore = ret1  # (5,65)*(65*10)得到(5,10)
                # 输出标准化，theloss是计算返回的损失
                theloss = softmax_loss.forward(thescore, ys)

        # print(thescore.shape,len(self.ret[0][0]),len(self.ret[1][0]))
    def backward(self, xs, relu,softmax_loss):
        # 反向传播
        ret=self.ret
        dw=self.d_dense #函数末尾再把值赋给d_dense
        dense_num = len(self.dense)
        w = self.dense.copy()
        for i in range(dense_num):
            j = dense_num-1 - i

            if j==dense_num-1:
                dscore = softmax_loss.backward()
                #获得最后一个系数矩阵的梯度
                dw[j]=np.dot(ret[j-1].T, dscore)
                dret = np.dot(dscore, w[j].T)
            else:
                dret0 = self.relu[j].backward(dret)
                # 反向传播得到的dret0要减去一列
                dret0 = np.delete(dret0, -1, axis=1)
                if j==0:
                    dw[j]=np.dot(xs.T, dret0)
                    # print('dw {} is {}'.format(j, dw[j].shape))
                    # dw.append(np.dot(xs.T, dret0))
                else:
                    #ret的数目比w矩阵少一个，所以要 j减一
                    dw[j]=np.dot(ret[j-1].T, dret0)
                    # print('dw {} is {}'.format(j,dw[j].shape))
                    # dw.append(np.dot(ret[j-1].T, dret0))
                dret=np.dot(dret0, w[j].T)
                # print(dret.shape,dret0.shape,w[j].shape)

        self.d_dense = dw

    def update_dense(self):
        length=len(self.dense)
        for i in range(length):
            self.dense[i] -= self.d_dense[i] * self.learning_rate


    def add_dense(self, w_num):
        cnt=len(self.dense) #系数矩阵w个数
        if cnt==0:
            w = np.random.rand(784, w_num) * math.sqrt(2.0 / (784 * w_num));
            b = np.zeros(w_num)
            w = np.insert(w, 784, b, axis=0)
            self.dense.append(w)
        else:
            last_w_num=len(self.dense[-1][0])
            w = np.random.rand(last_w_num, w_num) * math.sqrt(2.0 / (last_w_num * w_num));
            b = np.zeros(w_num)
            w = np.insert(w, last_w_num, b, axis=0)
            self.dense.append(w)
        # print(cnt)
        self.d_dense=list(np.zeros(cnt+1))
        self.ret=list(np.zeros(cnt))
        # 最后一层用不到relu，但多定义一个也没关系
        self.relu.append(dt.Relu())

    def train_net(self):
        relu = dt.Relu()
        softmax_loss = dt.SoftmaxWithLoss()

        for epoch in range(self.training_epochs):  # training_epochs
            if epoch >= 7:
                self.learning_rate=0.05
            if epoch >= 15:
                self.learning_rate = 0.02
            for step in range(self.total_step):
                # 每循环一次数据集，便输出一次loss、accuracy
                xs = self.train_x[step * self.batch_size:(step + 1) * self.batch_size]
                ys = self.train_y[step * self.batch_size:(step + 1) * self.batch_size]

                self.forward(xs,ys,relu,softmax_loss)
                self.backward(xs,relu,softmax_loss)
                self.update_dense()

            accuracy,loss = testm.test_model(self)
            # loss=eval_loss(w,xs,ys)
            print('epoch {} : the accuracy is {:.2f}% , the loss is {:.4f}'
                  .format(epoch+1, 100 * accuracy, loss))
