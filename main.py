import tensorflow as tf
import numpy as np
import test_module as testm
import neural_net as net
import matplotlib.pyplot as plt

def init_para():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # 预处理
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)
    # 归一化
    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0
    # 数据预处理，末尾都加一列“1”
    arr_one1 = np.ones(60000)
    arr_one2 = np.ones(10000)
    train_x = np.insert(train_images, 784, arr_one1, axis=1)
    train_y = train_labels
    test_x = np.insert(test_images, 784, arr_one2, axis=1)
    test_y = test_labels

    return train_x,train_y,test_x,test_y

def main():
    #获取mnist数据集
    (train_x, train_y, test_x, test_y) = init_para()
    #初始化网络参数
    network = net.nn(train_num=60000,
                     training_epochs=20,
                     batch_size=50,
                     lerning_rate = 0.1,   #训练过程中减小
                     train_x=train_x,
                     train_y=train_y,
                     test_x=test_x,
                     test_y=test_y
                    )
    #定义网络层级
    network.add_dense(100)
    network.add_dense(64)
    network.add_dense(32)
    network.add_dense(10)   #最后一层，输出层
    # 开始训练#
    print('***Start training***')
    network.train_net()
    #前10个样例的测试展示
    print('***Start predicting***')
    testm.predict(network)

if __name__ == '__main__':
    main()
