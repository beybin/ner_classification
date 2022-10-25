import os

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay, extend_with_piecewise_linear_lr
from bert4keras.layers import ConditionalRandomField
from bert4keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, Lambda, LSTM, Conv1D, Concatenate, \
    MultiHeadAttention, Embedding
from bert4keras.models import Model
from bert4keras.backend import K, keras
# from keras.utils import plot_model
from sklearn.metrics import cohen_kappa_score
# from optimizers import AdamW
from bert4keras.optimizers import AdaFactor

from utils import crossentropy_with_rdrop, adversarial_training, crossentropy_with_rdrop_1
from config import *


class Trainer:
    def __init__(self, mode='bert'):
        self.crf = ConditionalRandomField(lr_multiplier=crf_lr_multiplier, name='crf')
        # trans = K.eval(self.crf.trans)
        # 重计算
        # os.environ['RECOMPUTE'] = '1'
        self.mode = mode
        if mode == BERT:
            bert_dir = os.path.join(bert_path, 'chinese_wwm_ext_L-12_H-768_A-12')
            checkpoint_path = os.path.join(bert_dir, 'bert_model.ckpt')
        elif mode == ROBERTA:
            bert_dir = os.path.join(bert_path, 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
            # bert_dir = os.path.join(bert_path, 'chinese_roberta_wwm_large_ext_L_24_H_1024_A_16')
            checkpoint_path = os.path.join(bert_dir, 'bert_model.ckpt')
        else:
            bert_dir = os.path.join(bert_path, 'chinese_macbert_base')
            checkpoint_path = os.path.join(bert_dir, 'chinese_macbert_base.ckpt')

        config_path = os.path.join(bert_dir, 'bert_config.json')
        dict_path = os.path.join(bert_dir, 'vocab.txt')

        # 建立分词器
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.bert = build_transformer_model(
            config_path,
            checkpoint_path,
            return_keras_model=False,
            # dropout_rate=0.3
        )

        output1 = self.bert.model.output
        output = Dense(64, activation='relu')(output1)
        output = Dropout(0.1)(output)
        output = Dense(num_labels)(output)
        output_1 = self.crf(output)

        output2 = Lambda(lambda x: x[:, 0], name='CLS-token')(output1)
        output2 = Dense(64, activation='relu')(output2)
        output2 = Dropout(0.1)(output2)
        output_2 = Dense(
            units=num_classes,
            activation='softmax',
            kernel_initializer=self.bert.initializer,
            name="classify"
        )(output2)

        model = Model(self.bert.model.input, [output_1, output_2])
        model.summary()
        # plot_model(model, show_shapes=True, show_layer_names=False)

        optimizer = Adam(learning_rate=bert_lr[self.mode])
        # AdaFactor时，batch_size（32）和bert_lr（5e-4，或e-3级别）都要大一点
        # optimizer = AdaFactor(learning_rate=bert_lr[self.mode], beta1=0.9, min_dim_size_to_factor=10 ** 6)

        # from kappa import CohenKappa
        # kappa = CohenKappa(num_classes=3)

        model.compile(
            # loss={"crf": self.crf.sparse_loss, "classify": "sparse_categorical_crossentropy"},
            loss={"crf": self.crf.sparse_loss, "classify": crossentropy_with_rdrop},
            optimizer=optimizer,
            metrics={'crf': [self.crf.sparse_accuracy], 'classify': ['sparse_categorical_accuracy']},
            # metrics={'crf': [self.crf.sparse_accuracy], 'classify': [kappa]},
            loss_weights={'crf': 1, 'classify': 0.005}
        )

        # 写好函数后，启用对抗训练只需要一行代码
        # adversarial_training(model, 'Embedding-Token', 0.5)
        self.model = model
