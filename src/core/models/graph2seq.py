import random
import string
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.common import EncoderRNN, DecoderRNN, dropout
from ..layers.attention import *
from ..layers.graphs import GraphNN
from ..utils.generic_utils import to_cuda, create_mask
from ..utils.constants import VERY_SMALL_NUMBER


class Graph2SeqOutput(object):

  def __init__(self, encoder_outputs, encoder_state, decoded_tokens, \
          loss=0, loss_value=0, enc_attn_weights=None, ptr_probs=None):
    self.encoder_outputs = encoder_outputs
    self.encoder_state = encoder_state
    self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
    self.loss = loss  # scalar
    self.loss_value = loss_value  # float value, excluding coverage loss
    self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
    self.ptr_probs = ptr_probs  # (out seq len, batch size)


class Graph2Seq(nn.Module):

  def __init__(self, config, word_embedding, word_vocab):
    """
    :param word_vocab: mainly for info about special tokens and word_vocab size
    :param config: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the graph2seq model; its encoder and decoder will be created automatically.
    """
    super(Graph2Seq, self).__init__()
    self.name = 'Graph2Seq'
    self.device = config['device']
    self.word_dropout = config['word_dropout']
    self.edge_dropout = config['edge_dropout']
    self.bert_dropout = config['bert_dropout']
    self.word_vocab = word_vocab
    self.vocab_size = len(word_vocab)
    self.f_case = config['f_case']
    self.f_pos = config['f_pos']
    self.f_ner = config['f_ner']
    self.f_freq = config['f_freq']
    self.f_dep = config['f_dep']
    self.f_ans = config['f_ans']
    self.dan_type = config.get('dan_type', 'all')
    self.max_dec_steps = config['max_dec_steps']
    self.rnn_type = config['rnn_type']
    self.enc_attn = config['enc_attn']
    self.enc_attn_cover = config['enc_attn_cover']
    self.dec_attn = config['dec_attn']
    self.pointer = config['pointer']
    self.pointer_loss_ratio = config['pointer_loss_ratio']
    self.cover_loss = config['cover_loss']
    self.cover_func = config['cover_func']
    self.message_function = config['message_function']
    self.use_bert = config['use_bert']
    self.use_bert_weight = config['use_bert_weight']
    self.use_bert_gamma = config['use_bert_gamma']
    self.finetune_bert = config.get('finetune_bert', None)
    bert_dim = (config['bert_dim'] if self.use_bert else 0)

    enc_hidden_size = config['rnn_size']
    if config['dec_hidden_size']:
      dec_hidden_size = config['dec_hidden_size']
      if self.rnn_type == 'lstm':
        self.enc_dec_adapter = nn.ModuleList([nn.Linear(enc_hidden_size, dec_hidden_size) for _ in range(2)])
      else:
        self.enc_dec_adapter = nn.Linear(enc_hidden_size, dec_hidden_size)
    else:
      dec_hidden_size = enc_hidden_size
      self.enc_dec_adapter = None

    enc_input_dim = config['word_embed_dim']
    self.word_embed = word_embedding
    if config['fix_word_embed']:
      print('[ Fix word embeddings ]')
      for param in self.word_embed.parameters():
        param.requires_grad = False

    self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
    if self.f_case:
      self.case_embed = nn.Embedding(3, config['case_embed_dim'], padding_idx=0)
      enc_input_dim += config['case_embed_dim']
    if self.f_pos:
      self.pos_embed = nn.Embedding(config['num_features_f_pos'], config['pos_embed_dim'], padding_idx=0)
      enc_input_dim += config['pos_embed_dim']
    if self.f_ner:
      self.ner_embed = nn.Embedding(config['num_features_f_ner'], config['ner_embed_dim'], padding_idx=0)
      enc_input_dim += config['ner_embed_dim']
    if self.f_freq:
      self.freq_embed = nn.Embedding(4, config['freq_embed_dim'], padding_idx=0)
      enc_input_dim += config['freq_embed_dim']
    if self.f_dep:
      self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
      enc_input_dim += config['edge_embed_dim']
    if self.f_ans and self.dan_type in ('all', 'word'):
      enc_input_dim += config['word_embed_dim']
    if self.use_bert:
      enc_input_dim += config['bert_dim']


    if self.use_bert and self.use_bert_weight:
      num_bert_layers = config['bert_layer_indexes'][1] - config['bert_layer_indexes'][0]
      self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, num_bert_layers)))
      if self.use_bert_gamma:
          self.gamma_bert_layers = nn.Parameter(nn.init.constant_(torch.Tensor(1, 1), 1.))


    config['gl_input_size'] = enc_input_dim
    self.ctx_rnn_encoder = EncoderRNN(enc_input_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                              rnn_dropout=config['enc_rnn_dropout'], device=self.device)


    # Deep answer alignment
    if self.f_ans:
      if self.dan_type in ('all', 'word'):
        self.ctx2ans_attn_l1 = Context2AnswerAttention(config['word_embed_dim'], config['hidden_size'])

      if self.dan_type in ('all', 'hidden'):
        self.ans_rnn_encoder = EncoderRNN(config['word_embed_dim'] + bert_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                                  rnn_dropout=config['enc_rnn_dropout'], device=self.device)
        self.ctx2ans_attn_l2 = Context2AnswerAttention(config['word_embed_dim'] + config['hidden_size'] + bert_dim, config['hidden_size'])

        self.ctx_rnn_encoder_l2 = EncoderRNN(2 * enc_hidden_size, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                          rnn_dropout=config['enc_rnn_dropout'], device=self.device)

      print('[ Using Deep Answer Alignment Network: {} ]'.format(self.dan_type))

    self.graph_encoder = GraphNN(config)
    self.decoder = DecoderRNN(self.vocab_size, config['word_embed_dim'], dec_hidden_size, rnn_type=self.rnn_type,
                              enc_attn=config['enc_attn'], dec_attn=config['dec_attn'],
                              pointer=config['pointer'], out_embed_size=config['out_embed_size'],
                              tied_embedding=self.word_embed if config['tie_embed'] else None,
                              in_drop=config['dec_in_dropout'], rnn_drop=config['dec_rnn_dropout'],
                              out_drop=config['dec_out_dropout'], enc_hidden_size=enc_hidden_size, device=self.device)


  def filter_oov(self, tensor, ext_vocab_size):
    """Replace any OOV index in `tensor` with UNK"""
    if ext_vocab_size and ext_vocab_size > self.vocab_size:
      result = tensor.clone()
      result[tensor >= self.vocab_size] = self.word_vocab.UNK
      return result
    return tensor

  def get_coverage_vector(self, enc_attn_weights):
    """Combine the past attention weights into one vector"""
    if self.cover_func == 'max':
      coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
    elif self.cover_func == 'sum':
      coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
    else:
      raise ValueError('Unrecognized cover_func: ' + self.cover_func)
    return coverage_vector

  def forward(self, ex, target_tensor=None, criterion=None, criterion_reduction=True, criterion_nll_only=False, \
              rl_loss=False, *, forcing_ratio=0, partial_forcing=True, \
              ext_vocab_size=None, sample=False, saved_out: Graph2SeqOutput=None, \
              visualize: bool=None, include_cover_loss: bool=False) -> Graph2SeqOutput:
    """
    :param input_tensor: tensor of word indices, (batch size, src seq len)
    :param target_tensor: tensor of word indices, (batch size, tgt seq len)
    :param input_lengths: see explanation in `EncoderRNN`
    :param criterion: the loss function; if set, loss will be returned
    :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
    :param partial_forcing: see explanation in `Params` (training only)
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                   of greedily selecting the token of the highest probability at each step
    :param saved_out: the output of this function in a previous run; if set, the encoding step will
                      be skipped and we reuse the encoder states saved in this object
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

    Run the graph2seq model for training or testing.
    """
    input_tensor = ex['context']
    input_lengths = ex['context_lens']

    batch_size, input_length = input_tensor.shape
    input_mask = create_mask(input_lengths, input_length, self.device)

    log_prob = not (sample or self.decoder.pointer)  # don't apply log too soon in these cases
    if visualize is None:
      visualize = criterion is None
    if visualize and not (self.enc_attn or self.pointer):
      visualize = False  # nothing to visualize

    if target_tensor is None:
      target_length = self.max_dec_steps
      target_mask = None
    else:
      target_tensor = target_tensor.transpose(1, 0)
      target_length = target_tensor.size(0)
      target_mask = create_mask(ex['target_lens'], target_length, self.device)


    if forcing_ratio == 1:
      # if fully teacher-forced, it may be possible to eliminate the for-loop over decoder steps
      # for generality, this optimization is not investigated
      use_teacher_forcing = True
    elif forcing_ratio > 0:
      if partial_forcing:
        use_teacher_forcing = None  # decide later individually in each step
      else:
        use_teacher_forcing = random.random() < forcing_ratio
    else:
      use_teacher_forcing = False

    if saved_out:  # reuse encoder states of a previous run
      encoder_outputs = saved_out.encoder_outputs
      encoder_state = saved_out.encoder_state
      assert input_length == encoder_outputs.size(0)
      assert batch_size == encoder_outputs.size(1)
    else:  # run the encoder
      # encoder_embedded: (batch size, input len, embed size)
      encoder_embedded = self.word_embed(self.filter_oov(input_tensor, ext_vocab_size))
      encoder_embedded = dropout(encoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)

      enc_input_cat = [encoder_embedded]
      if self.f_case:
        case_features = self.case_embed(ex['context_case'])
        enc_input_cat.append(case_features)
      if self.f_pos:
        pos_features = self.pos_embed(ex['context_pos'])
        enc_input_cat.append(pos_features)
      if self.f_ner:
        ner_features = self.ner_embed(ex['context_ner'])
        enc_input_cat.append(ner_features)
      if self.f_freq:
        freq_features = self.freq_embed(ex['context_freq'])
        enc_input_cat.append(freq_features)
      if self.f_dep:
        dep_features = self.edge_embed(ex['context_dep'])
        enc_input_cat.append(dep_features)

      if self.f_ans:
        answer_tensor = ex['answers']
        answer_lengths = ex['answer_lens']
        ans_mask = create_mask(answer_lengths, answer_tensor.size(1), self.device)
        ans_embedded = self.word_embed(self.filter_oov(answer_tensor, ext_vocab_size))
        ans_embedded = dropout(ans_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
        enc_answer_cat = [ans_embedded]
        if self.dan_type in ('all', 'word'):
          # Align answer info to passage at the word level
          ctx_aware_ans_emb = self.ctx2ans_attn_l1(encoder_embedded, ans_embedded, ans_embedded, ans_mask)
          enc_input_cat.append(ctx_aware_ans_emb)


      if self.use_bert:
        context_bert = ex['context_bert']
        if not self.finetune_bert:
          assert context_bert.requires_grad == False
        if self.use_bert_weight:
          weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
          if self.use_bert_gamma:
            weights_bert_layers = weights_bert_layers * self.gamma_bert_layers

          context_bert = torch.mm(weights_bert_layers, context_bert.view(context_bert.size(0), -1)).view(context_bert.shape[1:])
          context_bert = dropout(context_bert, self.bert_dropout, shared_axes=[-2], training=self.training)
          enc_input_cat.append(context_bert)

          if self.f_ans and self.dan_type in ('all', 'hidden'):
            answer_bert = ex['answer_bert']
            if not self.finetune_bert:
              assert answer_bert.requires_grad == False

            answer_bert = torch.mm(weights_bert_layers, answer_bert.view(answer_bert.size(0), -1)).view(answer_bert.shape[1:])
            answer_bert = dropout(answer_bert, self.bert_dropout, shared_axes=[-2], training=self.training)
            enc_answer_cat.append(answer_bert)

      raw_input_vec = torch.cat(enc_input_cat, -1)
      encoder_outputs = self.ctx_rnn_encoder(raw_input_vec, input_lengths)[0].transpose(0, 1)

      if self.f_ans and self.dan_type in ('all', 'hidden'):
        # Align answer info to passage at the hidden state level
        enc_answer_cat = torch.cat(enc_answer_cat, -1)
        ans_encoder_outputs = self.ans_rnn_encoder(enc_answer_cat, answer_lengths)[0].transpose(0, 1)

        enc_cat_l2 = torch.cat([encoder_embedded, encoder_outputs], -1)
        ans_cat_l2 = torch.cat([ans_embedded, ans_encoder_outputs], -1)

        if self.use_bert:
          enc_cat_l2 = torch.cat([enc_cat_l2, context_bert], -1)
          ans_cat_l2 = torch.cat([ans_cat_l2, answer_bert], -1)

        ctx_aware_ans_emb = self.ctx2ans_attn_l2(enc_cat_l2, \
                              ans_cat_l2, ans_encoder_outputs, ans_mask)
        encoder_outputs = self.ctx_rnn_encoder_l2(torch.cat([encoder_outputs, ctx_aware_ans_emb], -1), \
                              input_lengths)[0].transpose(0, 1)


      input_graphs = ex['context_graphs']
      if self.message_function == 'edge_mm':
        edge_vec = input_graphs['edge_features']
      else:
        edge_vec = self.edge_embed(input_graphs['edge_features'])

      node_embedding, graph_embedding = self.graph_encoder(encoder_outputs, \
                  edge_vec, (input_graphs['node2edge'], input_graphs['edge2node']), \
                  node_mask=input_mask, raw_node_vec=raw_input_vec)
      encoder_outputs = node_embedding
      encoder_state = (graph_embedding, graph_embedding) if self.rnn_type == 'lstm' else graph_embedding


    # initialize return values
    r = Graph2SeqOutput(encoder_outputs, encoder_state,
                      torch.zeros(target_length, batch_size, dtype=torch.long))
    if visualize:
      r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
      if self.pointer:
        r.ptr_probs = torch.zeros(target_length, batch_size)

    if self.enc_dec_adapter is None:
      decoder_state = encoder_state
    else:
      if self.rnn_type == 'lstm':
        decoder_state = tuple([self.enc_dec_adapter[i](x) for i, x in enumerate(encoder_state)])
      else:
        decoder_state = self.enc_dec_adapter(encoder_state)
    decoder_hiddens = []
    enc_attn_weights = []


    enc_context = None
    dec_prob_ptr_tensor = []
    decoder_input = to_cuda(torch.tensor([self.word_vocab.SOS] * batch_size), self.device)
    for di in range(target_length):
      decoder_embedded = self.word_embed(self.filter_oov(decoder_input, ext_vocab_size))
      decoder_embedded = dropout(decoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
      if enc_attn_weights:
        coverage_vector = self.get_coverage_vector(enc_attn_weights)
      else:
        coverage_vector = None
      decoder_output, decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = \
        self.decoder(decoder_embedded, decoder_state, encoder_outputs,
                     torch.cat(decoder_hiddens) if decoder_hiddens else None, coverage_vector,
                     input_mask=input_mask,
                     encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size,
                     log_prob=log_prob,
                     prev_enc_context=enc_context)
      dec_prob_ptr_tensor.append(dec_prob_ptr)
      if self.dec_attn:
        decoder_hiddens.append(decoder_state[0] if self.rnn_type == 'lstm' else decoder_state)

      # save the decoded tokens
      if not sample:
        _, top_idx = decoder_output.data.topk(1)  # top_idx shape: (batch size, k=1)
      else:
        prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
        top_idx = torch.multinomial(prob_distribution, 1)
      top_idx = top_idx.squeeze(1).detach()  # detach from history as input
      r.decoded_tokens[di] = top_idx


      # decide the next input
      if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
        decoder_input = target_tensor[di]  # teacher forcing
      else:
        decoder_input = top_idx

      # compute loss
      if criterion:
        if target_tensor is None:
          gold_standard = top_idx  # for sampling
        else:
          gold_standard = target_tensor[di] if not rl_loss else decoder_input
        if not log_prob:
          decoder_output = torch.log(decoder_output + VERY_SMALL_NUMBER)  # necessary for NLLLoss

        if criterion_reduction:
          nll_loss = criterion(decoder_output, gold_standard)
          r.loss += nll_loss
          r.loss_value += nll_loss.item()
        else:
          nll_loss = F.nll_loss(decoder_output, gold_standard, ignore_index=self.word_vocab.PAD, reduction='none')
          r.loss += nll_loss
          r.loss_value += nll_loss


      # update attention history and compute coverage loss
      if self.enc_attn_cover or (criterion and self.cover_loss > 0):
        if not criterion_nll_only and coverage_vector is not None and criterion and self.cover_loss > 0:
          if criterion_reduction:
            coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * self.cover_loss
            r.loss += coverage_loss
            if include_cover_loss: r.loss_value += coverage_loss.item()
          else:
            coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn), dim=-1) * self.cover_loss
            r.loss += coverage_loss
            if include_cover_loss: r.loss_value += coverage_loss

        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
      # save data for visualization
      if visualize:
        r.enc_attn_weights[di] = dec_enc_attn.data
        if self.pointer:
          r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data


    # compute pointer network loss
    if not criterion_nll_only and criterion and self.pointer_loss_ratio > 0 and target_tensor is not None:
      dec_prob_ptr_tensor = torch.cat(dec_prob_ptr_tensor, -1)
      pointer_loss = F.binary_cross_entropy(dec_prob_ptr_tensor, ex['target_copied'], reduction='none')
      if criterion_reduction:
        pointer_loss = torch.sum(pointer_loss * target_mask) / batch_size * self.pointer_loss_ratio
        r.loss += pointer_loss
        r.loss_value += pointer_loss.item()
      else:
        pointer_loss = torch.sum(pointer_loss * target_mask, dim=-1) * self.pointer_loss_ratio
        r.loss += pointer_loss
        r.loss_value += pointer_loss
    return r
