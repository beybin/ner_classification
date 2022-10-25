import math

import keras.losses
from bert4keras.backend import K, search_layer
import numpy as np
from tqdm import tqdm
from keras.losses import kullback_leibler_divergence as kld
from sklearn.metrics import cohen_kappa_score


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


def evaluate(tester, data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    ner_pred_list = []
    true_list = []
    pred_list = []
    for dd in tqdm(data):
        # d: ((text, bio_label), class_label)
        d = dd[0]
        class_label = dd[1]

        text = ''.join([i[0] for i in d])
        ner_res, class_res = tester.test(text)
        R = set([(text[res[0]:res[1]], res[2]) for res in ner_res])
        ner_pred_list.append(R)

        T = set([tuple(i) for i in d if i[1] != 'O'])

        X += len(R & T)
        Y += len(R)
        Z += len(T)

        # 分类正确数
        true_list.append(class_label)
        pred_list.append(class_res)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    tc_kappa = cohen_kappa_score(true_list, pred_list)
    score = 0.5 * f1 + 0.5 * tc_kappa
    return f1, precision, recall, tc_kappa, score, ner_pred_list, pred_list


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


from tensorflow.python.ops import array_ops
import tensorflow as tf


def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # 将超出指定范围的数强制变为边界值keras.backend.clip(x, min_value, max_value)
    # print(y_pred)
    # print(y_pred.shape)
    y_true = K.cast(y_true, tf.int32)  # 转化int
    num_classes = array_ops.shape(y_pred)[array_ops.rank(y_pred) - 1]
    y_true = K.one_hot(y_true, num_classes=num_classes)
    y_true = K.cast(y_true, tf.float32)  # 再转回来
    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))


def crossentropy_with_rdrop_1(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = loss_with_gradient_penalty(y_true, y_pred)
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


# cw = {2: 5159, 0: 635, 1: 228}
# print(create_class_weight(cw))
#
# from sklearn.utils.class_weight import compute_class_weight
#
# label = [0] * 9 + [1] * 1 + [2, 2]
# classes = [0, 1, 2]
# weight = compute_class_weight(class_weight='balanced', classes=classes, y=label)
# print(weight)

# train = {2: 5159, 0: 635, 1: 228}
# print(636/6022, 230/6022, 5156/6022)
# test = {2: 1286, 0: 160, 1: 60}
# print(159/1506, 58/1506, 1289/1506)
