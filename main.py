import csv
import gc
import sklearn.model_selection
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from data_util import load_data
from tester import Tester, TesterAll
from trainer import Trainer
from utils import evaluate, create_class_weight
from sklearn.utils.class_weight import compute_class_weight
from config import train_data_path, test_data_path, mode_list, epochs, batch_size, max_seq_len, label2id

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
print(tf.__version__, tf.test.is_gpu_available())


# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# try:
#     gpus = tf.config.list_physical_devices('GPU')
#     # tf.config.experimental.set_memory_growth(gpus[0], True)
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[1],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)]
#     )
# except Exception as e:
#     print('Invalid device or cannot modify virtual devices once initialized.')


class Evaluate(keras.callbacks.Callback):
    def __init__(self, tester, valid_data, model_name):
        # super().__init__()
        # self.best_val_f1 = 0
        self.best_val_score = 0
        self.tester = tester
        self.validation_data = valid_data
        self.model_name = model_name
        # self.best_val_precision = 0
        # self.best_val_recall = 0

    def on_epoch_end(self, epoch, logs=None):
        # trans = K.eval(CRF.trans)
        f1, precision, recall, tc_kappa, score, _, _ = evaluate(self.tester, self.validation_data)
        # 保存最优
        if score >= self.best_val_score:
            self.best_val_score = score
            # self.best_val_precision = precision
            # self.best_val_recall = recall
            self.model.save_weights(f'{self.model_name}')
        print('valid: f1: %.5f, precision: %.5f, recall: %.5f, tc_kappa: %.5f, score: %.5f, best score: %.5f\n' % (
            f1, precision, recall, tc_kappa, score, self.best_val_score))


class data_generator(DataGenerator):
    """
    数据生成器
    """

    def __init__(self, data, tokenizer):
        super().__init__(data, batch_size)
        self.tokenizer = tokenizer

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels, batch_labels_class = [], [], [], []
        for is_end, (item, class_label) in self.sample(random):
            token_ids, labels = [self.tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < max_seq_len:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [self.tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)

            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                batch_labels_class.append([class_label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_labels_class = sequence_padding(batch_labels_class)
                yield [batch_token_ids, batch_segment_ids], [batch_labels, batch_labels_class]
                batch_token_ids, batch_segment_ids, batch_labels, batch_labels_class = [], [], [], []


def get_random_data(data):
    labels = [d[1] for d in data]
    train_data, valid_data, _, _ = sklearn.model_selection.train_test_split(data, labels, train_size=0.8,
                                                                            stratify=labels, random_state=42)
    label_list = [d[1] for d in train_data]
    class_weight = Counter(label_list)
    # class_weight = create_class_weight(Counter(label_list))
    # class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(label_list), y=label_list)
    # class_weight = dict(zip(range(3), class_weight))
    return train_data, valid_data, class_weight


def train(data):
    X_train = [i[0] for i in data]
    Y_train = [i[1] for i in data]

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1, mode='min')
    mode = 'roberta'

    stratifiedKFolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    count = 0
    for (trn_idx, val_idx) in stratifiedKFolds.split(X_train, Y_train):
        count += 1
        print(f'count:{count}')
        #         if count < 1:
        #            continue
        #         K.clear_session()
        trainer = Trainer(mode=mode)
        tokenizer = trainer.tokenizer
        tester = Tester(trainer, testing=False)
        model = trainer.model

        model_name = f'split_{count}.weights'

        train_data = [(X_train[ti], Y_train[ti]) for ti in trn_idx]
        valid_data = [(X_train[ti], Y_train[ti]) for ti in val_idx]

        # train_data, valid_data, class_weight = get_random_data(data)
        label_list = [d[1] for d in train_data]
        class_weight = Counter(label_list)
        train_generator = data_generator(data=train_data, tokenizer=tokenizer)
        evaluator = Evaluate(tester=tester, valid_data=valid_data, model_name=model_name)

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator, early_stopping],
            # {0: 635, 1: 228, 2: 5159}
            class_weight={'classify': class_weight}
        )

        del model
        gc.collect()
        K.clear_session()


def fix_train_label(entities, entity_count):
    entity_list = []
    for entity in entities:
        entity_list.append((entity['start_pos'], entity['end_pos'], entity['label_type']))

    if entity_list == list(entity_count.keys()):
        return entity_list

    fix_res = []
    for i in range(len(entity_list)):
        entity = entity_list[i]
        if entity in entity_count:
            fix_res.append(entity)

    for k, v in entity_count.items():
        if entity_count[k] == 3 and k not in entity_list:
            fix_res.append(k)
    return fix_res


def evaluate_text_single(test=True):
    mode = 'roberta'
    if test:
        path = test_data_path
    else:
        path = train_data_path
    for count in range(1, 6):
        tester = Tester(Trainer(mode), testing=True, weghts_path=f'split_{count}.weights')
        if test:
            wf_path = f'res_{count}.csv'
        else:
            wf_path = f'res_train_{count}.csv'
        print(wf_path)
        wf = csv.writer(open(wf_path, 'w', encoding='utf8', newline=''))
        for row in csv.reader(open(path, 'r', encoding='utf8')):
            if test:
                line_no, line = row
            else:
                line_no, line, bio_label, class_true = row
            if line_no == 'id':
                continue
            print(f'count:{count}, row:{int(line_no) + 1}')

            result_list, class_pred_line = tester.test_text(line)
            wf.writerow([line_no, result_list, class_pred_line])

    if not test:
        return
    wf = csv.writer(open(f'res.csv', 'w', encoding='utf8', newline=''))
    wf.writerow(['id', 'BIO_anno', 'class'])
    rf_list = []
    for count in range(1, 6):
        rf = []
        for r in csv.reader(open(f'res_{count}.csv', encoding='utf8')):
            rf.append(r)
        rf_list.append(rf)

    for row in csv.reader(open(test_data_path, 'r', encoding='utf8')):
        line_no, line = row
        if line_no == 'id':
            continue
        print('count:', int(line_no) + 1)
        index = int(line_no)
        entity_count = {}
        class_count = []
        pred_list = []
        for rf in rf_list:
            _, res, class_pred = rf[index]
            for pred in eval(res):
                entity_count[pred] = entity_count.get(pred, 0) + 1
            class_count.append(int(class_pred))

        for k in entity_count:
            if entity_count[k] > 2:
                pred_list.append(k)

        label_list = ['O' for _ in range(len(line))]
        for sp, ep, en_type in pred_list:
            for i in range(sp, ep):
                if i == sp:
                    label_list[i] = 'B-' + en_type
                else:
                    label_list[i] = 'I-' + en_type

        wf.writerow([line_no, ' '.join(label_list), max(class_count, key=class_count.count)])


def evaluate_text(file_path):
    '''
    加载多模型预测NER，并写入文件
    :param file_path:
    :param predict_type:为1时预测训练数据，为2时预测验证/测试数据，为3时预测未标注数据
    :return:
    '''
    tester_list = []
    for mode in mode_list:
        tester_list.append(Tester(Trainer(mode), testing=True))

    wf = csv.writer(open(file_path, 'w', encoding='utf8', newline=''))
    wf.writerow(['id', 'BIO_anno', 'class'])
    f = csv.reader(open(test_data_path, 'r', encoding='utf8'))

    for row in f:
        line_no, line = row
        if line_no == 'id':
            continue
        print('count:', int(line_no) + 1)
        # if int(line_no) + 1 < 16:
        #     continue

        entity_count = {}
        class_count = []
        pred_list = []

        for tester in tester_list:
            result_list, class_pred_line = tester.test_text(line)
            for pred in result_list:
                entity_count[pred] = entity_count.get(pred, 0) + 1
            class_count.append(class_pred_line)

        if len(mode_list) > 2:
            for k in entity_count:
                if entity_count[k] > 1:
                    pred_list.append(k)
        else:
            pred_list = result_list

        label_list = ['O' for _ in range(len(line))]
        for sp, ep, en_type in pred_list:
            for i in range(sp, ep):
                if i == sp:
                    label_list[i] = 'B-' + en_type
                else:
                    label_list[i] = 'I-' + en_type

        wf.writerow([line_no, ' '.join(label_list), max(class_count, key=class_count.count)])


if __name__ == '__main__':
    train_flag = False

    if train_flag:
        data = load_data()
        train(data=data)
    else:
        # 测试多模型融合效果
        #         data = load_data()
        #         train_data, valid_data, class_weight = get_random_data(data)
        #         tester_all = TesterAll(valid_data)
        #         tester_all.test()

        # 预测训练数据、验证/测试数据、未标注数据
        # evaluate_text('res.csv')
        evaluate_text_single(test=True)
