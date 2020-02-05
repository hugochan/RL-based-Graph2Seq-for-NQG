import random
import string
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.common import EncoderRNN, DecoderRNN, dropout
from ..layers.attention import *
from ..utils.generic_utils import to_cuda, create_mask
from ..utils.constants import VERY_SMALL_NUMBER


class Seq2SeqOutput(object):

  def __init__(self, encoder_outputs, encoder_state, decoded_tokens, \
          loss=0, loss_value=0, enc_attn_weights=None, ptr_probs=None):
    self.encoder_outputs = encoder_outputs
    self.encoder_state = encoder_state
    self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
    self.loss = loss  # scalar
    self.loss_value = loss_value  # float value, excluding coverage loss
    self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
    self.ptr_probs = ptr_probs  # (out seq len, batch size)


class Seq2Seq(nn.Module):
  '''BERT feature is not implemented yet.'''
  def __init__(self, config, word_embedding, word_vocab):
    """
    :param word_vocab: mainly for info about special tokens and word_vocab size
    :param config: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the seq2seq model; its encoder and decoder will be created automatically.
    """
    super(Seq2Seq, self).__init__()
    self.name = 'Seq2Seq'
    self.device = config['device']
    self.word_dropout = config['word_dropout']
    self.word_vocab = word_vocab
    self.vocab_size = len(word_vocab)
    self.f_case = config['f_case']
    self.f_pos = config['f_pos']
    self.f_ner = config['f_ner']
    self.f_freq = config['f_freq']
    self.f_dep = config['f_dep']
    self.f_ans = config['f_ans']
    self.max_dec_steps = config['max_dec_steps']
    self.rnn_type = config['rnn_type']
    self.enc_attn = config['enc_attn']
    self.enc_attn_cover = config['enc_attn_cover']
    self.dec_attn = config['dec_attn']
    self.pointer = config['pointer']
    self.pointer_loss_ratio = config['pointer_loss_ratio']
    self.cover_loss = config['cover_loss']
    self.cover_func = config['cover_func']
    self.use_bert = config['use_bert']

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
    if self.f_ans:
      enc_input_dim += config['word_embed_dim']


    self.ctx_rnn_encoder = EncoderRNN(enc_input_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                              rnn_dropout=config['enc_rnn_dropout'], device=self.device)
    self.ctx_rnn_encoder_l2 = EncoderRNN(2 * enc_hidden_size, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                              rnn_dropout=config['enc_rnn_dropout'], device=self.device)
    self.ans_rnn_encoder = EncoderRNN(config['word_embed_dim'], enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                              rnn_dropout=config['enc_rnn_dropout'], device=self.device)

    self.decoder = DecoderRNN(self.vocab_size, config['word_embed_dim'], dec_hidden_size, rnn_type=self.rnn_type,
                              enc_attn=config['enc_attn'], dec_attn=config['dec_attn'],
                              pointer=config['pointer'], out_embed_size=config['out_embed_size'],
                              tied_embedding=self.word_embed if config['tie_embed'] else None,
                              in_drop=config['dec_in_dropout'], rnn_drop=config['dec_rnn_dropout'],
                              out_drop=config['dec_out_dropout'], enc_hidden_size=enc_hidden_size, device=self.device)

    # Answer alignment
    self.ctx2ans_attn_l1 = Context2AnswerAttention(config['word_embed_dim'], config['hidden_size'])
    self.ctx2ans_attn_l2 = Context2AnswerAttention(config['word_embed_dim'] + config['hidden_size'], config['hidden_size'])

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

  def forward(self, ex, target_tensor=None, criterion=None, criterion_reduction=True, \
              criterion_nll_only=False, rl_loss=False, *, forcing_ratio=0, partial_forcing=True, \
              ext_vocab_size=None, sample=False, saved_out: Seq2SeqOutput=None, \
              visualize: bool=None, include_cover_loss: bool=False) -> Seq2SeqOutput:
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
    input_graphs = ex['context_graphs']

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
        # Align answer info to passage at the word level
        answer_tensor = ex['answers']
        answer_lengths = ex['answer_lens']
        ans_mask = create_mask(answer_lengths, answer_tensor.size(1), self.device)
        ans_embedded = self.word_embed(self.filter_oov(answer_tensor, ext_vocab_size))
        ans_embedded = dropout(ans_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
        ctx_aware_ans_emb = self.ctx2ans_attn_l1(encoder_embedded, ans_embedded, ans_embedded, ans_mask)
        enc_input_cat.append(ctx_aware_ans_emb)

      encoder_outputs, encoder_state = self.ctx_rnn_encoder(torch.cat(enc_input_cat, -1), input_lengths)
      if self.f_ans:
        # Align answer info to passage at the hidden state level
        encoder_outputs = encoder_outputs.transpose(0, 1)
        ans_encoder_outputs = self.ans_rnn_encoder(ans_embedded, answer_lengths)[0].transpose(0, 1)
        ctx_aware_ans_emb = self.ctx2ans_attn_l2(torch.cat([encoder_embedded, encoder_outputs], -1), \
                              torch.cat([ans_embedded, ans_encoder_outputs], -1), ans_encoder_outputs, ans_mask)
        encoder_outputs, encoder_state = self.ctx_rnn_encoder_l2(torch.cat([encoder_outputs, ctx_aware_ans_emb], -1), \
                              input_lengths)


    # initialize return values
    r = Seq2SeqOutput(encoder_outputs, encoder_state,
                      torch.zeros(target_length, batch_size, dtype=torch.long))
    if visualize:
      r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
      if self.pointer:
        r.ptr_probs = torch.zeros(target_length, batch_size)

    decoder_input = to_cuda(torch.tensor([self.word_vocab.SOS] * batch_size), self.device)
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
          gold_standard = target_tensor[di]
        if not log_prob:
          decoder_output = torch.log(decoder_output + VERY_SMALL_NUMBER)  # necessary for NLLLoss
        nll_loss = criterion(decoder_output, gold_standard)
        r.loss += nll_loss
        r.loss_value += nll_loss.item()

      # update attention history and compute coverage loss
      if self.enc_attn_cover or (criterion and self.cover_loss > 0):
        if coverage_vector is not None and criterion and self.cover_loss > 0:
          coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * self.cover_loss
          r.loss += coverage_loss
          if include_cover_loss: r.loss_value += coverage_loss.item()
        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
      # save data for visualization
      if visualize:
        r.enc_attn_weights[di] = dec_enc_attn.data
        if self.pointer:
          r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data


    if criterion and self.pointer_loss_ratio > 0 and target_tensor is not None:
      dec_prob_ptr_tensor = torch.cat(dec_prob_ptr_tensor, -1)
      pointer_loss = F.binary_cross_entropy(dec_prob_ptr_tensor, ex['target_copied'], reduction='none')
      pointer_loss = torch.sum(pointer_loss * target_mask) / batch_size * self.pointer_loss_ratio
      r.loss += pointer_loss
      r.loss_value += pointer_loss.item()
    return r
