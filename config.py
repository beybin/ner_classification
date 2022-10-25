bert_path = 'bert'

# 训练数据、测试数据
train_data_path = 'data/train_data_public-modify.csv'
test_data_path = 'data/test_public.csv'

# 模型参数
max_seq_len = 128
epochs = 5
batch_size = 8
# bert_layers = 12
crf_lr_multiplier = 1  # 必要时扩大CRF层的学习率

# 实体类别映射
labels = ['BANK', 'PRODUCT', 'COMMENTS_N', 'COMMENTS_ADJ']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

num_classes = 3

# 句子切分符号
# split_str = '！!？?。；;，,：:、'
split_str = '！!？?。，,'

# 滑动窗口大小，预测超过最大长度时使用
padding_size = 10
window_size = max_seq_len - 2 - 2 * padding_size


BERT = 'bert'
ROBERTA = 'roberta'
# ELECTRA = 'electra'
MACBERT = 'macbert'
mode_list = [ROBERTA]

# bert_layers越小，学习率应该要越大
# bert_lr = 2e-5
bert_lr = {BERT: 1e-5, ROBERTA: 2e-5, MACBERT: 1e-5}
