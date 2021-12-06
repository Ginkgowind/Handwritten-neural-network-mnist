import numpy as np

def softmax(score):
    length = score.shape[0]

    max_list = np.amax(score, axis=1)
    # 减去每行最大值
    for i in range(length):
        score[i] -= max_list[i]

    array_exp = np.exp(score)
    # 每行指数总和的列表
    sum_list = np.sum(array_exp, axis=1)

    for i in range(length):
        array_exp[i] = np.divide(array_exp[i], sum_list[i])
    array_ret = array_exp.copy()

    return array_ret  # 返回(10000,10)的概率列表

def loss_EVM(y, t):
    # y,预测概率 t,实际标签  均为(batch_size,10)
    batch_size = y.shape[0]
    ten_one = one_hot(t)
    a = np.array(y)
    b = np.array(ten_one)
    c = a - b
    ret = c ** 2
    loss = np.sum(ret) / batch_size

    return loss

def one_hot(t):
    # t为一个列表 返回一个二维独热编码矩阵
    length = len(t)
    result = np.zeros([length, 10])
    for i in range(length):
        max_index = t[i]  # 独热编码 即将标签转化为列表，方便与实际值比较
        result[i] = [j == max_index for j in range(10)]

    return result

print(one_hot([1,2,3]))