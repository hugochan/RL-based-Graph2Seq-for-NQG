# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import re
import random
import io
import torch
import numpy as np
from scipy.sparse import *
from collections import Counter, defaultdict
from .timer import Timer

from .bert_utils import *
from . import padding_utils
from . import constants


def vectorize_input(batch, config, bert_model, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch.sent1_word)

    context = torch.LongTensor(batch.sent1_word)
    context_lens = torch.LongTensor(batch.sent1_length)

    questions = torch.LongTensor(batch.sent2_word)
    question_lens = torch.LongTensor(batch.sent2_length)

    if batch.has_sent3:
        answers = torch.LongTensor(batch.sent3_word)
        answer_lens = torch.LongTensor(batch.sent3_length)


    # Extract features from pretrained BERT models
    if config['use_bert']:
        with torch.set_grad_enabled(False):
            layer_indexes = list(range(config['bert_layer_indexes'][0], config['bert_layer_indexes'][1]))

            # Passage words
            max_d_len = batch.sent1_length.max().item()
            max_bert_d_num_chunks = max([len(ex_bert_d) for ex_bert_d in batch.sent1_bert])
            max_bert_d_len = max([len(bert_d.input_ids) for ex_bert_d in batch.sent1_bert for bert_d in ex_bert_d])
            bert_xd = torch.LongTensor(batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
            bert_xd_mask = torch.LongTensor(batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
            for i, ex_bert_d in enumerate(batch.sent1_bert): # Example level
                for j, bert_d in enumerate(ex_bert_d): # Chunk level
                    bert_xd[i, j, :len(bert_d.input_ids)].copy_(torch.LongTensor(bert_d.input_ids))
                    bert_xd_mask[i, j, :len(bert_d.input_mask)].copy_(torch.LongTensor(bert_d.input_mask))
            if device:
                bert_xd = bert_xd.to(device)
                bert_xd_mask = bert_xd_mask.to(device)
            all_encoder_layers, _ = bert_model(bert_xd.view(-1, bert_xd.size(-1)), token_type_ids=None, attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)))
            torch.cuda.empty_cache()
            all_encoder_layers = torch.stack([x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers], 0).detach()
            all_encoder_layers = all_encoder_layers[layer_indexes]
            bert_context_f = extract_bert_hidden_states(all_encoder_layers, max_d_len, batch.sent1_bert, weighted_avg=config['use_bert_weight'])
            torch.cuda.empty_cache()


            # Answer words
            max_a_len = batch.sent3_length.max().item()
            max_bert_a_num_chunks = max([len(ex_bert_a) for ex_bert_a in batch.sent3_bert])
            max_bert_a_len = max([len(bert_a.input_ids) for ex_bert_a in batch.sent3_bert for bert_a in ex_bert_a])
            bert_xa = torch.LongTensor(batch_size, max_bert_a_num_chunks, max_bert_a_len).fill_(0)
            bert_xa_mask = torch.LongTensor(batch_size, max_bert_a_num_chunks, max_bert_a_len).fill_(0)
            for i, ex_bert_a in enumerate(batch.sent3_bert): # Example level
                for j, bert_a in enumerate(ex_bert_a): # Chunk level
                    bert_xa[i, j, :len(bert_a.input_ids)].copy_(torch.LongTensor(bert_a.input_ids))
                    bert_xa_mask[i, j, :len(bert_a.input_mask)].copy_(torch.LongTensor(bert_a.input_mask))
            if device:
                bert_xa = bert_xa.to(device)
                bert_xa_mask = bert_xa_mask.to(device)
            all_encoder_layers, _ = bert_model(bert_xa.view(-1, bert_xa.size(-1)), token_type_ids=None, attention_mask=bert_xa_mask.view(-1, bert_xa_mask.size(-1)))
            torch.cuda.empty_cache()
            all_encoder_layers = torch.stack([x.view(bert_xa.shape + (-1,)) for x in all_encoder_layers], 0).detach()
            all_encoder_layers = all_encoder_layers[layer_indexes]
            bert_answer_f = extract_bert_hidden_states(all_encoder_layers, max_a_len, batch.sent3_bert, weighted_avg=config['use_bert_weight'])
            torch.cuda.empty_cache()


    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'context': context.to(device) if device else context,
                   'context_lens': context_lens.to(device) if device else context_lens,
                   'context_graphs': batch.sent1_graph,
                   'targets': questions.to(device) if device else questions,
                   'target_lens': question_lens.to(device) if device else question_lens,
                   'target_src': batch.sent2_src,
                   'oov_dict': batch.oov_dict}

        if batch.has_sent3:
            example['answers'] = answers.to(device) if device else answers
            example['answer_lens'] = answer_lens.to(device) if device else answer_lens
        if config['f_case']:
            context_case = torch.LongTensor(batch.sent1_case)
            example['context_case'] = context_case.to(device) if device else context_case
        if config['f_pos']:
            context_pos = torch.LongTensor(batch.sent1_POS)
            example['context_pos'] = context_pos.to(device) if device else context_pos
        if config['f_ner']:
            context_ner = torch.LongTensor(batch.sent1_NER)
            example['context_ner'] = context_ner.to(device) if device else context_ner
        if config['f_freq']:
            context_freq = torch.LongTensor(batch.sent1_freq)
            example['context_freq'] = context_freq.to(device) if device else context_freq
        if config['f_dep']:
            context_dep = torch.LongTensor(batch.sent1_dep)
            example['context_dep'] = context_dep.to(device) if device else context_dep
        if config['use_bert']:
            example['context_bert'] = bert_context_f
            example['answer_bert'] = bert_answer_f

        if config['pointer_loss_ratio'] > 0:
            target_copied = torch.Tensor(batch.sent2_copied)
            example['target_copied'] = target_copied.to(device) if device else target_copied
        return example

def prepare_datasets(config):
    if config['trainset'] is not None:
        train_set, train_src_len, train_tgt_len, train_ans_len = read_all_GenerationDatasets(config['trainset'], isLower=True)
        print('# of training examples: {}'.format(len(train_set)))
        print('Training source sentence length: {}'.format(train_src_len))
        print('Training target sentence length: {}'.format(train_tgt_len))
        print('Training answer sentence length: {}'.format(train_ans_len))
    else:
        train_set = None

    if config['devset'] is not None:
        dev_set, dev_src_len, dev_tgt_len, dev_ans_len = read_all_GenerationDatasets(config['devset'], isLower=True)
        print('# of dev examples: {}'.format(len(dev_set)))
        print('Dev source sentence length: {}'.format(dev_src_len))
        print('Dev target sentence length: {}'.format(dev_tgt_len))
        print('Dev answer sentence length: {}'.format(dev_ans_len))
    else:
        dev_set = None

    if config['testset'] is not None:
        test_set, test_src_len, test_tgt_len, test_ans_len = read_all_GenerationDatasets(config['testset'], isLower=True)
        print('# of testing examples: {}'.format(len(test_set)))
        print('Testing source sentence length: {}'.format(test_src_len))
        print('Testing target sentence length: {}'.format(test_tgt_len))
        print('Testing answer sentence length: {}'.format(test_ans_len))
    else:
        test_set = None
    return {'train': train_set, 'dev': dev_set, 'test': test_set}


def read_all_GenerationDatasets(inpath, isLower=True):
    with open(inpath) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')
    all_instances = []
    src_len = []
    tgt_len = []
    ans_len = []
    for instance in dataset:
        ID_num = None
        if 'id' in instance: ID_num = instance['id']

        text1 = instance['annotation1']['toks'] if 'annotation1' in instance else instance['text1']
        if text1 == "": continue
        annotation1 = instance['annotation1'] if 'annotation1' in instance else None
        sent1 = QASentence(text1, annotation1, ID_num=ID_num, isLower=isLower)
        src_len.append(sent1.get_length())

        text2 = instance['annotation2']['toks'] if 'annotation2' in instance else instance['text2']
        if text2 == "": continue
        annotation2 = instance['annotation2'] if 'annotation2' in instance else None
        sent2 = QASentence(text2, annotation2, ID_num=ID_num, isLower=isLower, end_sym=constants._EOS_TOKEN)
        tgt_len.append(sent2.get_length()) # text2 is the sequence to be generated

        sent3 = None
        if 'text3' in instance or 'annotation3' in instance:
            text3 = instance['annotation3']['toks'] if 'annotation3' in instance else instance['text3']
            annotation3 = instance['annotation3'] if 'annotation3' in instance else None
            sent3 = QASentence(text3, annotation3, ID_num=ID_num, isLower=isLower)
            ans_len.append(sent3.get_length())
        all_instances.append((sent1, sent2, sent3))

    src_len_stats = {'min': np.min(src_len), 'max': np.max(src_len), 'mean': np.mean(src_len)}
    tgt_len_stats = {'min': np.min(tgt_len), 'max': np.max(tgt_len), 'mean': np.mean(tgt_len)}
    if len(ans_len) > 0:
        ans_len_stats = {'min': np.min(ans_len), 'max': np.max(ans_len), 'mean': np.mean(ans_len)}
    else:
        ans_len_stats = None
    return all_instances, src_len_stats, tgt_len_stats, ans_len_stats

class QADataStream(object):
    def __init__(self, all_instances, word_vocab, edge_vocab, POS_vocab=None, NER_vocab=None, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1, ext_vocab=False, bert_tokenizer=None):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda question: (question[0].get_length(), question[1].get_length()))
        else:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = QAQuestionBatch(cur_instances, config, word_vocab, edge_vocab,
                    POS_vocab=POS_vocab, NER_vocab=NER_vocab, ext_vocab=ext_vocab, bert_tokenizer=bert_tokenizer)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]

class QAQuestionBatch(object):
    def __init__(self, instances, config, word_vocab, edge_vocab, POS_vocab=None, NER_vocab=None, ext_vocab=False, bert_tokenizer=None):
        self.instances = instances
        self.batch_size = len(instances)
        self.oov_dict = None # out-of-vocabulary dict

        self.has_sent3 = False
        if instances[0][2] is not None: self.has_sent3 = True

        # Create word representation and length
        self.sent1_word = [] # [batch_size, sent1_len]
        self.sent1_length = [] # [batch_size]

        self.sent2_src = []
        self.sent2_word = [] # [batch_size, sent2_len]
        self.sent2_length = [] # [batch_size]

        if self.has_sent3:
            self.sent3_word = [] # [batch_size, sent3_len]
            self.sent3_length = [] # [batch_size]

        if config['f_case']:
            self.sent1_case = [] # [batch_size, sent1_len]

        if config['f_pos']:
            self.sent1_POS = [] # [batch_size, sent1_len]

        if config['f_ner']:
            self.sent1_NER = [] # [batch_size, sent1_len]

        if config['f_freq']:
            self.sent1_freq = [] # [batch_size, sent1_len]

        if config['f_dep']:
            self.sent1_dep = [] # [batch_size, sent1_len]

        if config['pointer_loss_ratio'] > 0:
            self.sent2_copied = [] # [batch_size, sent1_len]

        if config['use_bert']:
            self.sent1_bert = []
            if self.has_sent3:
                self.sent3_bert = []

        if ext_vocab:
            base_oov_idx = len(word_vocab)
            self.oov_dict = OOVDict(base_oov_idx)

        batch_graph = []
        for i, (sent1, sent2, sent3) in enumerate(instances):
            sent1_idx = []
            for word in sent1.words:
                idx = word_vocab.getIndex(word)
                if ext_vocab and idx == word_vocab.UNK:
                    idx = self.oov_dict.add_word(i, word)
                sent1_idx.append(idx)
            self.sent1_word.append(sent1_idx)
            self.sent1_length.append(sent1.get_length())
            batch_graph.append(sent1.graph)

            if config['f_freq']:
                self.sent1_freq.append([1 if x <= config['high_freq_rank'] else (2 if x < config['low_freq_rank'] else 3) for x in sent1_idx])

            if config['f_dep']:
                tmp = {}
                for val in sent1.graph['g_adj'].values():
                    for each in val:
                        if each['edge'] == 'neigh':
                            continue
                        tmp[each['node']] = edge_vocab.getIndex(each['edge'])
                dep_idx = []
                for idx in range(len(sent1_idx)):
                    dep_idx.append(tmp.get(idx, edge_vocab.getIndex('neigh')))

                self.sent1_dep.append(dep_idx)

            if config['use_bert']:
                bert_sent1_features = convert_text_to_bert_features(sent1.words, bert_tokenizer, config['bert_max_seq_len'], config['bert_doc_stride'])
                self.sent1_bert.append(bert_sent1_features)

            sent2_idx = []
            sent2_copied_idx = []
            for word in sent2.words:
                idx = word_vocab.getIndex(word)
                if ext_vocab and idx == word_vocab.UNK:
                    idx = self.oov_dict.word2index.get((i, word), idx)
                sent2_idx.append(idx)

                if config['pointer_loss_ratio'] > 0:
                    # check if the word is copied using some heuristics, i.e., shared low-frequency non-stopwords
                    if is_copied(word, sent1.words) and idx > config['high_freq_rank']:
                        sent2_copied_idx.append(1)
                    else:
                        sent2_copied_idx.append(0)

            self.sent2_word.append(sent2_idx)
            self.sent2_src.append(sent2.src)
            self.sent2_length.append(sent2.get_length())
            if config['pointer_loss_ratio'] > 0:
                self.sent2_copied.append(sent2_copied_idx)

            if self.has_sent3:
                self.sent3_word.append([word_vocab.getIndex(word) for word in sent3.words])
                self.sent3_length.append(sent3.get_length())

                if config['use_bert']:
                    bert_sent3_features = convert_text_to_bert_features(sent3.words, bert_tokenizer, config['bert_max_seq_len'], config['bert_doc_stride'])
                    self.sent3_bert.append(bert_sent3_features)

            if config['f_case']:
                self.sent1_case.append(sent1.CASEs)

            if config['f_pos']:
                self.sent1_POS.append(POS_vocab.to_index_sequence(sent1.POSs))

            if config['f_ner']:
                self.sent1_NER.append(NER_vocab.to_index_sequence(sent1.NERs))


        # Build graph
        batch_graphs = cons_batch_graph(batch_graph)
        self.sent1_graph = vectorize_batch_graph(batch_graphs, edge_vocab, config)
        self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)

        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        self.sent2_word = padding_utils.pad_2d_vals_no_size(self.sent2_word)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)
        if config['pointer_loss_ratio'] > 0:
            self.sent2_copied = padding_utils.pad_2d_vals_no_size(self.sent2_copied)

        if self.has_sent3:
            self.sent3_word = padding_utils.pad_2d_vals_no_size(self.sent3_word)
            self.sent3_length = np.array(self.sent3_length, dtype=np.int32)

        if config['f_case']:
            self.sent1_case = padding_utils.pad_2d_vals_no_size(self.sent1_case)

        if config['f_pos']:
            self.sent1_POS = padding_utils.pad_2d_vals_no_size(self.sent1_POS)

        if config['f_ner']:
            self.sent1_NER = padding_utils.pad_2d_vals_no_size(self.sent1_NER)

        if config['f_freq']:
            self.sent1_freq = padding_utils.pad_2d_vals_no_size(self.sent1_freq)

        if config['f_dep']:
            self.sent1_dep = padding_utils.pad_2d_vals_no_size(self.sent1_dep)


class QASentence(object):
    def __init__(self, tokText, annotation=None, ID_num=None, isLower=False, end_sym=None):
        self.CASEs = []
        self.POSs = ''
        self.NERs = ''

        self.src = tokText.lower() if isLower else tokText
        self.tokText = tokText
        # it's the answer sequence
        if end_sym != None:
            self.tokText += ' ' + end_sym

        for each in re.split("\\s+", self.tokText):
            self.CASEs.append(1 if each.islower() else 2)

        if isLower:
            self.tokText = self.tokText.lower()
        self.words = re.split("\\s+", self.tokText)

        if annotation != None:
            self.POSs = annotation.get('POSs', '')
            self.NERs = annotation.get('NERs', '')
            self.graph = annotation.get('graph', None)
        self.length = len(self.words)
        self.ID_num = ID_num

        self.index_convered = False
        self.chunk_starts = None

    def get_length(self):
        return self.length

def cons_batch_graph(graphs):
    num_nodes = max([len(g['g_features']) for g in graphs])
    num_edges = max([g['num_edges'] for g in graphs])

    batch_edges = []
    batch_node2edge = []
    batch_edge2node = []
    for g in graphs:
        edges = {}
        node2edge = lil_matrix(np.zeros((num_edges, num_nodes)), dtype=np.float32)
        edge2node = lil_matrix(np.zeros((num_nodes, num_edges)), dtype=np.float32)
        edge_index = 0
        for node1, value in g['g_adj'].items():
            node1 = int(node1)
            for each in value:
                node2 = int(each['node'])
                if node1 == node2: # Ignore self-loops for now
                    continue
                edges[edge_index] = each['edge']
                node2edge[edge_index, node2] = 1
                edge2node[node1, edge_index] = 1
                edge_index += 1
        batch_edges.append(edges)
        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)
    batch_graphs = {'max_num_edges': num_edges,
                    'edge_features': batch_edges,
                    'node2edge': batch_node2edge,
                    'edge2node': batch_edge2node
                    }
    return batch_graphs

def vectorize_batch_graph(graph, edge_vocab, config):
    # vectorize the graph
    edge_features = []
    for edges in graph['edge_features']:
        edges_v = []
        for idx in range(len(edges)):
            edges_v.append(edge_vocab.getIndex(edges[idx]))
        for _ in range(graph['max_num_edges'] - len(edges_v)):
            edges_v.append(edge_vocab.PAD)
        edge_features.append(edges_v)

    edge_features = torch.LongTensor(np.array(edge_features))

    gv = {'edge_features': edge_features.to(config['device']) if config['device'] else edge_features,
          'node2edge': graph['node2edge'],
          'edge2node': graph['edge2node']
          }
    return gv

class OOVDict(object):
    def __init__(self, base_oov_idx):
        self.word2index = {}  # type: Dict[Tuple[int, str], int]
        self.index2word = {}  # type: Dict[Tuple[int, int], str]
        self.next_index = {}  # type: Dict[int, int]
        self.base_oov_idx = base_oov_idx
        self.ext_vocab_size = base_oov_idx

    def add_word(self, idx_in_batch, word) -> int:
        key = (idx_in_batch, word)
        index = self.word2index.get(key, None)
        if index is not None: return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index


from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
def is_copied(word, target):
    if word in target and word not in stopWords:
        return True
    else:
        return False
