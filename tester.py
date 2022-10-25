from bert4keras.backend import K
from bert4keras.snippets import ViterbiDecoder

from utils import evaluate
from config import *
from trainer import Trainer
from data_util import get_split_list
from sklearn.metrics import cohen_kappa_score


class Tester:
    def __init__(self, trainer, testing=False, weghts_path=''):
        self.tokenizer = trainer.tokenizer
        self.viterbi_decoder = ViterbiDecoder(trans=K.eval(trainer.crf.trans), starts=[0], ends=[0])
        # self.trans = K.eval(model.crf.trans)
        self.model = trainer.model

        # 测试需加载训练好的权重文件
        if testing:
            self.model.load_weights(weghts_path)

    def test(self, text):
        '''
        预测一条长度不超过512的数据
        :param text:
        :return: 由(sp,ep,en_type)组成的列表
        '''
        tokens = self.tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        # 预测的crf得分（维度为(1,9,9)）和情感分类概率（维度为(1,3)，如：[[0.00199844 0.00163868 0.99636286]]）
        pred = self.model.predict([[token_ids], [segment_ids]])
        nodes = pred[0][0]
        class_pred = pred[1].argmax(axis=1)[0]

        labels = self.viterbi_decoder.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        # return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities]
        return [(mapping[w[0]][0], mapping[w[-1]][-1] + 1, l) for w, l in entities], class_pred

    def test_eval(self, text):
        dic = {'originalText': text, 'entities': []}
        res = self.test_text(text)
        # label_list = ['O' for _ in range(len(text))]
        for sp, ep, en_type in res:
            dic['entities'].append({'start_pos': sp, 'end_pos': ep, 'label_type': en_type})

        return dic

    def fix_result(self, result):
        '''
        修正result的结果，返回
        先按首升序，再按尾降序
        如果首大于尾，则没问题
        如果后首小于前尾，说明有重叠，判断两个实体的长度，选取较长的那个
        '''
        if not result:
            return []
        new_result = []
        temp = result[0]
        result = sorted(result, key=lambda x: (x[0], -x[1]))
        for i, r in enumerate(result):
            if not r[2]:
                continue
            if i == 0:
                continue
            if r[0] > temp[1]:  # 当前的头大于之前的尾
                new_result.append(temp)
                temp = r
            # else: # 当前的头小于之前的尾，代表有重叠
            # 因为已经排序过，所以长的包含短的时候永远先选择长的
        if temp[2]:
            new_result.append(temp)
        return new_result

    def test_text(self, text):
        '''
        预测一条数据，如果长度超过512，使用滑动窗口分多次预测
        :param text:
        :return: 由(sp,ep,en_type)组成的列表
        '''
        res, class_pred = self.test(text)
        if len(text) <= max_seq_len - 2:
            return res, class_pred
        else:
            result = []
            text_list = get_split_list(text)
            offset = 0
            for sent in text_list:
                if len(sent) < window_size + 2 * padding_size:
                    temp_res_list, _ = self.test(sent)
                    for res in temp_res_list:
                        result.append((res[0] + offset, res[1] + offset, res[2]))
                        if text[res[0] + offset:res[1] + offset] != sent[res[0]:res[1]]:
                            print(sent)

                    offset += len(sent) - 2 * padding_size
                else:
                    for idx in range(padding_size, len(sent) - padding_size, window_size):
                        tsent = sent[idx - padding_size: idx + window_size + padding_size]
                        temp_res_list, _ = self.test(tsent)
                        for res in temp_res_list:
                            result.append((res[0] + offset, res[1] + offset, res[2]))
                            if text[res[0] + offset:res[1] + offset] != tsent[res[0]:res[1]]:
                                print(sent)

                        offset += len(tsent) - 2 * padding_size
                offset += 2 * padding_size
            # 修正ner结果，加入最终结果列表
            result = self.fix_result(result)
            return result, class_pred


class TesterAll:
    def __init__(self, valid_data):
        self.valid_data = valid_data

        # 列表的元素是一条数据的结果
        self.true_list = [set([tuple(i) for i in d[0] if i[1] != 'O']) for d in valid_data]
        self.true_tc_list = [d[1] for d in valid_data]
        self.total_count = len(self.valid_data)

    def test_one(self, index):
        mode = mode_list[index]
        tester = Tester(Trainer(mode), testing=True)

        # ner_pred_list = []
        # X, Y, Z = 1e-10, 1e-10, 1e-10
        # tc_pred_list = []
        # for i, (d, class_label) in tqdm(enumerate(self.valid_data)):
        #     text = ''.join([i[0] for i in d])
        #     ner_pred, class_pred = tester.test(text)
        #     tc_pred_list.append(class_pred)
        #
        #     pred_set = set([(text[res[0]:res[1]], res[2]) for res in ner_pred])
        #     ner_pred_list.append(pred_set)
        #
        #     true_set = self.true_list[i]
        #
        #     X += len(pred_set & true_set)
        #     Y += len(pred_set)
        #     Z += len(true_set)
        #
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        # return f1, precision, recall, cohen_kappa_score(self.true_tc_list, tc_pred_list), ner_pred_list, tc_pred_list
        return evaluate(tester, self.valid_data)

    def test(self):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        f1, precision, recall, tc_kappa, score, pred_list1, tc_list1 = self.test_one(0)
        print('res:', f1, precision, recall, tc_kappa, score)

        f1, precision, recall, tc_kappa, score, pred_list2, tc_list2 = self.test_one(1)
        print('res:', f1, precision, recall, tc_kappa, score)

        f1, precision, recall, tc_kappam, score, pred_list3, tc_list3 = self.test_one(2)
        print('res:', f1, precision, recall, tc_kappa, score)

        tc_preds = []
        for i in range(self.total_count):
            entity_count = {}
            for pred in pred_list1[i]:
                entity_count[pred] = entity_count.get(pred, 0) + 1

            for pred in pred_list2[i]:
                entity_count[pred] = entity_count.get(pred, 0) + 1

            for pred in pred_list3[i]:
                entity_count[pred] = entity_count.get(pred, 0) + 1

            pred_set = set()
            for k in entity_count:
                if entity_count[k] > 1:
                    pred_set.add(k)

            # for pred in pred_set:
            #     if pred not in self.true_list[i]:
            #         print(pred)

            X += len(pred_set & self.true_list[i])
            Y += len(pred_set)
            Z += len(self.true_list[i])

            tc_pred_list = [tc_list1[i], tc_list2[i], tc_list3[i]]
            tc_pred = max(tc_pred_list, key=tc_pred_list.count)
            tc_preds.append(tc_pred)

            if self.true_tc_list[i] != tc_pred and tc_pred_list.count(tc_pred) > 1:
                text = ''.join([i[0] for i in self.valid_data[i][0]])
                print('eee:', self.true_tc_list[i], tc_pred, tc_pred_list, text[:128])

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        tc_kappa = cohen_kappa_score(self.true_tc_list, tc_preds)
        print('res:', f1, precision, recall, tc_kappa, 0.5 * f1 + 0.5 * tc_kappa)
