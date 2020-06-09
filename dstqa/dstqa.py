// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
// Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import pdb
import math
import logging
import os.path
import pickle
import random
from typing import Any, Dict, List
from overrides import overrides
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, ArrayField
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.dataset import Batch
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TokenEmbedder, TextFieldEmbedder, FeedForward, ScalarMix
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn import Activation
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import logsumexp
from allennlp.tools import squad_eval
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy
from allennlp.modules.elmo import batch_to_ids as elmo_batch_to_ids
from allennlp.modules.elmo import Elmo

from .accuracy import Accuracy
from . import dstqa_util

logger = logging.getLogger(__name__)


@Model.register("dstqa")
class DSTQA(Model):
  def __init__(self, vocab: Vocabulary,
             base_dim,
             loss_scale_by_num_values,
             use_pre_calc_elmo_embeddings,
             elmo_embedding_path,
             domain_slot_list_path,
             word_embeddings,
             token_indexers: Dict[str, TokenIndexer],
             text_field_embedder: TextFieldEmbedder,
             text_field_char_embedder: TextFieldEmbedder,
             symbol_embedder: TextFieldEmbedder,
             phrase_layer: Seq2SeqEncoder,
             class_prediction_layer: FeedForward,
             span_prediction_layer: FeedForward,
             span_start_encoder: FeedForward,
             span_end_encoder: FeedForward,
             span_label_predictor: FeedForward,
             initializer: InitializerApplicator,
             use_graph,
             bi_dropout: float = 0.2,
             dropout: float = 0.2) -> None:
    super().__init__(vocab)
    self._is_in_training_mode = False
    self._loss_scale_by_num_values = loss_scale_by_num_values
    self._use_pre_calc_elmo_embeddings = use_pre_calc_elmo_embeddings
    self._word_embeddings = word_embeddings
    self._is_use_elmo = True if self._word_embeddings == "elmo" else False
    self._is_use_graph = use_graph
    if self._is_use_elmo and use_pre_calc_elmo_embeddings:
      self._dialog_elmo_embeddings = self.load_elmo_embeddings(elmo_embedding_path)
      self._dialog_scalar_mix = ScalarMix(mixture_size = 3, trainable=True)

    self._domains, self._ds_id2text, self._ds_text2id, self.value_file_path, \
      self._ds_type, self._ds_use_value_list, num_ds_use_value, self._ds_masked \
                                                                        = self.read_domain_slot_list(domain_slot_list_path)
    self._value_id2text, self._value_text2id = self.load_value_list(domain_slot_list_path)
    self._span_id2text, self._class_id2text = dstqa_util.gen_id2text(self._ds_id2text, self._ds_type)
    self._token_indexers = token_indexers

    self._text_field_embedder = text_field_embedder
    self._text_field_char_embedder = text_field_char_embedder
    self._symbol_embedder = symbol_embedder

    self._ds_dialog_attention = LinearMatrixAttention(base_dim, base_dim, 'x,y,x*y')
    self._dialog_dsv_attention = LinearMatrixAttention(base_dim, base_dim, 'x,y,x*y')
    self._dsv_dialog_attention = LinearMatrixAttention(base_dim, base_dim, 'x,y,x*y')
    self._ds_attention = LinearMatrixAttention(base_dim, base_dim, 'x,y,x*y')
    self._dsv_attention = LinearMatrixAttention(base_dim, base_dim, 'x,y,x*y')
    self._agg_value = torch.nn.Linear(base_dim, base_dim)
    self._agg_nodes = torch.nn.Linear(base_dim, base_dim)
    self._graph_gamma = torch.nn.Linear(base_dim, 1)
    self._class_prediction_layer = class_prediction_layer
    self._span_prediction_layer = span_prediction_layer
    self._span_label_predictor = span_label_predictor
    self._span_start_encoder = span_start_encoder
    self._span_end_encoder = span_end_encoder
    self._phrase_layer = phrase_layer
    self._cross_entropy = CrossEntropyLoss(ignore_index=-1)
    self._accuracy = Accuracy(self._ds_id2text, self._ds_type)
    self._dropout = torch.nn.Dropout(dropout)
    self._bi_dropout = torch.nn.Dropout(bi_dropout)
    self._dropout2 = torch.nn.Dropout(0.1)
    self._sigmoid = torch.nn.Sigmoid()
    initializer(self)

  def load_elmo_embeddings(self, elmo_embedding_path):
    elmo_embeddings = {}
    for suffix in ["train", "dev", "test"]:
      with open(elmo_embedding_path + suffix, "rb") as fp:
        elmo_embeddings.update(pickle.load(fp))
    return elmo_embeddings

  def gen_utt_masks(self, turn_offset, batch_size, max_turn_count, max_dialog_len):
    masks = torch.arange(0, max_dialog_len).unsqueeze(0).unsqueeze(0).cuda()
    masks = masks.repeat(batch_size, max_turn_count, 1)
    repeated_turn_offset = turn_offset.unsqueeze(2).repeat(1, 1, max_dialog_len)
    masks = masks < repeated_turn_offset
    # two types of masks: (1) all previous and current utt are marked as 1, (2) only current utt are marked as 1
    bmasks = masks.clone().detach()
    bmasks =  (~bmasks)[:, :-1, :]
    cmasks = masks.clone().detach()
    cmasks[:, 1:, :] = cmasks[:, 1:, :] & bmasks
    return masks, cmasks

  def mix_dialog_embeddings(self, dialog_indices):
    dialog_embeddings = []
    max_dialog_len = 0
    for idx in dialog_indices:
      elmo_embeddings_cuda = []
      for v in self._dialog_elmo_embeddings[idx]:
        elmo_embeddings_cuda.append(v.cuda())
      dialog_embeddings.append(self._dialog_scalar_mix(elmo_embeddings_cuda))
      if max_dialog_len < dialog_embeddings[-1].size(0):
        max_dialog_len = dialog_embeddings[-1].size(0)
    for i, e in enumerate(dialog_embeddings):
      pad = torch.zeros(max_dialog_len - e.size(0), e.size(1)). cuda()
      dialog_embeddings[i] = torch.cat((e, pad), dim=0)
    dialog_embeddings = torch.stack(dialog_embeddings, dim=0)
    return dialog_embeddings

  def mask_time_step(self, dialogs, dialog_masks):
    batch_size, max_dialog_len, max_char_len = dialogs['token_characters'].size()
    masks = self._dropout2(torch.ones(batch_size, max_dialog_len))
    masks = masks < 0.5
    char_masked = torch.tensor([259, 260] + [0] * (max_char_len - 2)).cuda()
    char_padded = torch.tensor([0] * max_char_len).cuda()
    dialogs["token_characters"][masks] = char_masked
    dialogs["token_characters"][dialog_masks == 0] = char_padded
    
    if "tokens" in dialogs:
      dialogs["tokens"][masks] = 1 # 1 is the index for unknown
      dialogs["tokens"][dialog_masks == 0] = 0
    if "elmo" in dialogs:
      elmo_masked = torch.tensor([259, 260] + [261] * (50 - 2)).cuda()
      elmo_padded = torch.tensor([0] * 50).cuda()
      dialogs["elmo"][masks] = elmo_masked
      dialogs["elmo"][dialog_masks == 0] = elmo_padded

  def forward(self, dialogs, tags, utt_lens, exact_match, dialog_indices, epoch_num = None,
              labels=None, spans_start=None, spans_end=None, metadata=None, span_labels=None):
    self._is_in_training_mode = self.training
    # dialog embeddings
    batch_size, max_dialog_len, _ = dialogs['token_characters'].size()
    dialog_masks = util.get_text_field_mask(dialogs, num_wrapping_dims=0)
    self.mask_time_step(dialogs, dialog_masks)
    char_embedder_input = {'token_characters':dialogs['token_characters']}
    dialog_char_embeddings = self._text_field_char_embedder(char_embedder_input, num_wrapping_dims=0)
    if self._is_use_elmo:
      if self._use_pre_calc_elmo_embeddings == False:
        elmo_embedder_input = {'elmo':dialogs['elmo']}
        dialog_elmo_embeddings = self._text_field_embedder(elmo_embedder_input, num_wrapping_dims=0)
        dialog_embeddings = torch.cat((dialog_elmo_embeddings, dialog_char_embeddings), dim = 2)
      else:
        dialog_elmo_embeddings = self.mix_dialog_embeddings(dialog_indices)
        dialog_embeddings = torch.cat((dialog_elmo_embeddings, dialog_char_embeddings), dim=2)
    else:
      embedder_input = {'tokens':dialogs['tokens']}
      dialog_elmo_embeddings = self._text_field_embedder(embedder_input, num_wrapping_dims=0)
      dialog_embeddings = torch.cat((dialog_elmo_embeddings, dialog_char_embeddings), dim = 2)
    tag_embeddings = self._symbol_embedder(tags, num_wrapping_dims=0)
    turn_offset = torch.cumsum(utt_lens, dim=1)
    max_turn_count = utt_lens.size(1)
    context_masks, utt_masks = self.gen_utt_masks(turn_offset, batch_size, max_turn_count, max_dialog_len)

    # dsv embeddings
    ds_embeddings, v_embeddings = self.get_dsv_embeddings()

    # phrase layer
    merged_dialog_embeddings = torch.cat((dialog_embeddings, tag_embeddings, exact_match), dim=2)

    total_loss = 0.0
    predictions = []
    if self._is_in_training_mode == True: # # only train one domain per turn for GPU memory limits
      sampled_turn = random.choice(list(range(max_turn_count)))
    for turn_i in range(max_turn_count):
      predictions.append(({}, {}))
      if self._is_in_training_mode == True and self._is_use_graph == False:
        if turn_i != sampled_turn:
          continue
      if self._is_in_training_mode == True:
        if turn_i < sampled_turn:
          self.set_module_to_eval()
        if turn_i > sampled_turn: break
      # compute new domain slot embeddings
      attention_ds_embeddings = None
      if turn_i > 0 and self._is_use_graph:
        attention_ds_embeddings = self.ds_graph_embeddings(batch_size, predictions[turn_i - 1], ds_embeddings, v_embeddings)
      repeated_ds_embeddings = ds_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
      reduced_dialog_masks = self._phrase_layer(self._dropout(merged_dialog_embeddings), context_masks[:, turn_i, :])
      span_ds_i = 0
      for ds_i, ds_name in enumerate(self._ds_id2text):
        cur_repeated_ds_embeddings = repeated_ds_embeddings[:, ds_i, :].unsqueeze(1)
        cur_context_masks = context_masks[:, turn_i, :]
        if self._ds_type[ds_name] == "classification":
          cur_labels = labels[:, turn_i, ds_i]
          cur_v_embeddings = v_embeddings[ds_name]
          loss, prediction = self.forward_classification(ds_name, reduced_dialog_masks, cur_repeated_ds_embeddings, cur_v_embeddings, cur_context_masks, cur_labels, attention_ds_embeddings)
          predictions[turn_i][0][ds_name] = prediction
          if self._loss_scale_by_num_values:
            loss = loss * max(1.0, math.log(cur_v_embeddings.size(0)))
        elif self._ds_type[ds_name] == "span":
          cur_span_labels = span_labels[:, turn_i, span_ds_i]
          cur_spans_start = spans_start[:, turn_i, span_ds_i]
          cur_spans_end = spans_end[:, turn_i, span_ds_i]
          loss, prediction = self.forward_span(ds_name, reduced_dialog_masks, cur_repeated_ds_embeddings, cur_context_masks, cur_span_labels, cur_spans_start, cur_spans_end)
          predictions[turn_i][1][ds_name] = prediction
          span_ds_i += 1
        if self._is_in_training_mode == True and turn_i == sampled_turn:
          if not self._ds_masked[ds_name]:
            total_loss += loss
      if self._is_in_training_mode == True:
        if turn_i < sampled_turn:
          self.set_module_to_train()
    output = {}
    if self._is_in_training_mode == True:
      output["loss"] = total_loss
    output["predictions"] = predictions
    output["metadata"] = metadata
    return output

  def set_module_to_eval(self):
    self.eval()
    self._phrase_layer.eval()
    self._class_prediction_layer.eval()
    self._span_prediction_layer.eval()
    self._span_start_encoder.eval()
    self._span_end_encoder.eval()
    self._span_label_predictor.eval()
    torch.set_grad_enabled(False)

  def set_module_to_train(self):
    self.train()
    self._phrase_layer.train()
    self._class_prediction_layer.train()
    self._span_prediction_layer.train()
    self._span_start_encoder.train()
    self._span_end_encoder.train()
    self._span_label_predictor.train()
    torch.set_grad_enabled(True)

  def bi_att(self, dialog_embeddings, dsv_embeddings, context_masks):
    batch_size, max_dialog_len = context_masks.size()
    num_values = dsv_embeddings.size(1)
    dialog_dsv_similarity = self._dialog_dsv_attention(self._bi_dropout(dialog_embeddings), self._bi_dropout(dsv_embeddings))
    # attention on dsv
    dialog_dsv_att = util.masked_softmax(dialog_dsv_similarity.view(-1, num_values), None)
    dialog_dsv_att = dialog_dsv_att.view(batch_size, max_dialog_len, num_values)
    dialog_dsv = util.weighted_sum(dsv_embeddings, dialog_dsv_att)
    new_dialog_embeddings = dialog_embeddings + dialog_dsv

    # attention on dialog
    dsv_dialog_att = util.masked_softmax(dialog_dsv_similarity.transpose(1, 2).contiguous().view(-1, max_dialog_len), context_masks.unsqueeze(1).repeat(1,num_values,1).view(-1, max_dialog_len))
    dsv_dialog_att = dsv_dialog_att.view(batch_size, num_values, max_dialog_len)
    dsv_dialog = util.weighted_sum(dialog_embeddings, dsv_dialog_att)
    new_dsv_embeddings = dsv_embeddings + dsv_dialog
    return new_dialog_embeddings, new_dsv_embeddings

  def forward_classification(self, ds_name, dialog_repr, ds_embeddings, value_embeddings, context_masks, labels=None, attention_ds_embeddings=None):
    batch_size, max_dialog_len = context_masks.size()
    num_values = value_embeddings.size(0)
    repeated_dsv_embeddings = ds_embeddings.repeat(1, num_values, 1)
    repeated_dsv_embeddings += value_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
    dialog_repr, repeated_dsv_embeddings = self.bi_att(dialog_repr, repeated_dsv_embeddings, context_masks)

    ds_dialog_sim = self._ds_dialog_attention(self._bi_dropout(ds_embeddings), self._bi_dropout(dialog_repr))
    ds_dialog_att = util.masked_softmax(ds_dialog_sim.view(-1, max_dialog_len), context_masks.view(-1, max_dialog_len))
    ds_dialog_att = ds_dialog_att.view(batch_size, max_dialog_len)
    ds_dialog_repr = util.weighted_sum(dialog_repr, ds_dialog_att)
    
    if attention_ds_embeddings is not None:
      self_att_matrix = self._ds_attention(self._bi_dropout(ds_dialog_repr.unsqueeze(1)), attention_ds_embeddings)
      self_probs = util.masked_softmax(self_att_matrix, None)
      ret = util.weighted_sum(attention_ds_embeddings, self_probs).squeeze(1)
      gamma = torch.sigmoid(self._graph_gamma(ds_dialog_repr + ret))
      ds_dialog_repr = (1-gamma) * ds_dialog_repr + gamma * ret

    w = self._class_prediction_layer(self._bi_dropout(ds_dialog_repr)).unsqueeze(1)
    logits = torch.bmm(w, repeated_dsv_embeddings.transpose(1,2)).squeeze(1)
    prediction = torch.argmax(logits, dim=1)
    loss = self._cross_entropy(logits.view(-1, num_values), labels.view(-1))
    if labels is not None:
      self._accuracy.value_acc(ds_name, logits, labels, labels != -1)
    return loss, prediction

  def forward_span(self, ds_name, dialog_repr, repeated_ds_embeddings, context_masks, span_labels=None, spans_start = None, spans_end = None):
    batch_size, max_dialog_len = context_masks.size()
    ds_dialog_sim = self._ds_dialog_attention(self._dropout(repeated_ds_embeddings), self._dropout(dialog_repr))
    ds_dialog_att = util.masked_softmax(ds_dialog_sim.view(-1, max_dialog_len), context_masks.view(-1, max_dialog_len))
    ds_dialog_att = ds_dialog_att.view(batch_size, max_dialog_len)
    ds_dialog_repr = util.weighted_sum(dialog_repr, ds_dialog_att)
    ds_dialog_repr = ds_dialog_repr + repeated_ds_embeddings.squeeze(1)
    span_label_logits = self._span_label_predictor(F.relu(self._dropout(ds_dialog_repr)))
    span_label_prediction = torch.argmax(span_label_logits, dim=1)
    span_label_loss = 0.0
    if span_labels is not None:
      span_label_loss = self._cross_entropy(span_label_logits, span_labels) # loss averaged by #turn
      self._accuracy.span_label_acc(ds_name, span_label_logits, span_labels, span_labels != -1)
    loss = span_label_loss

    w = self._span_prediction_layer(self._dropout(ds_dialog_repr)).unsqueeze(1)
    span_start_repr = self._span_start_encoder(self._dropout(dialog_repr))
    span_start_logits = torch.bmm(w, span_start_repr.transpose(1,2)).squeeze(1)
    span_start_probs = util.masked_softmax(span_start_logits, context_masks)
    span_start_logits = util.replace_masked_values(span_start_logits, context_masks.to(dtype=torch.int8), -1e7)

    span_end_repr = self._span_end_encoder(self._dropout(span_start_repr))
    span_end_logits = torch.bmm(w, span_end_repr.transpose(1,2)).squeeze(1)
    span_end_probs = util.masked_softmax(span_end_logits, context_masks)
    span_end_logits = util.replace_masked_values(span_end_logits, context_masks.to(dtype=torch.int8), -1e7)

    best_span = self.get_best_span(span_start_logits, span_end_logits)
    best_span = best_span.view(batch_size, -1)

    spans_loss = 0.0
    if spans_start is not None:
      spans_loss = self._cross_entropy(span_start_logits, spans_start)
      self._accuracy.span_start_acc(ds_name, span_start_logits, spans_start, spans_start != -1)
      spans_loss += self._cross_entropy(span_end_logits, spans_end)
      self._accuracy.span_end_acc(ds_name, span_end_logits, spans_end, spans_end != -1)
    loss += spans_loss

    return loss, (span_label_prediction, best_span)

  @overrides
  def decode(self, output_dict):
      num_turns = len(output_dict["predictions"])
      class_output = []
      for t in range(num_turns):
        class_predictions = output_dict["predictions"][t][0]
        res = []
        for ds_name, pred in class_predictions.items():
          value = self._value_id2text[ds_name][pred.item()]
          res.append(ds_name+":"+value)
        class_output.append(res)

      span_output = []
      for t in range(num_turns):
        span_predictions = output_dict["predictions"][t][1]
        res = []
        for ds_name, pred in span_predictions.items():
          span_label = pred[0]
          if span_label == 0: value = "none"
          if span_label == 1: value = "dont care"
          if span_label == 2: 
            start, end = pred[1][0][0], pred[1][0][1]
            value = " ".join([output_dict["metadata"][0][i].text for i in range(start, end+1)])
            value = value.lower()
          res.append(ds_name+":" + value)
        span_output.append(res)
      # merge class output and span output
      output = []
      if len(span_output) != 0 and len(class_output) != 0:
        for x, y in zip(class_output, span_output):
          output.append(x + y)
      elif len(span_output) == 0:
        output = class_output
      elif len(class_output) == 0:
        output = span_output
      else:
        assert(False)
      output_dict["predicted_labels"] = [output]
      del output_dict["metadata"]
      del output_dict["predictions"]
      return output_dict

  def get_metrics(self, reset = False):
      acc = self._accuracy.get_metrics(reset)
      return acc

  def get_dsv_embeddings(self):
    def batch_to_id(batch: List[List[str]]):
      instances = []
      for b in batch:
        tokens = [Token(w) for w in b.split(" ")]
        field = TextField(tokens, self._token_indexers)
        instance = Instance({"b": field})
        instances.append(instance)
      dataset = Batch(instances)
      vocab = self.vocab
      dataset.index_instances(vocab)
      res = {}
      for k, v in dataset.as_tensor_dict()['b'].items():
        res[k] = v.cuda()
      return res
    ds_ids = batch_to_id(self._ds_id2text)
    if 'tokens' in ds_ids:
      elmo_embedder_input = {'tokens':ds_ids['tokens']}
    elif 'elmo' in ds_ids:
      elmo_embedder_input = {'elmo':ds_ids['elmo']}
    ds_elmo_embeddings = self._text_field_embedder(elmo_embedder_input, num_wrapping_dims=0).sum(1)
    char_embedder_input = {'token_characters':ds_ids['token_characters']}
    ds_char_embeddings = self._text_field_char_embedder(char_embedder_input, num_wrapping_dims=0).sum(1)
    ds_embeddings = torch.cat((ds_elmo_embeddings, ds_char_embeddings), dim=1)
    ds_masks = util.get_text_field_mask(ds_ids, num_wrapping_dims=0).sum(1).float()
    ds_embeddings = ds_embeddings / ds_masks.unsqueeze(1).repeat(1, ds_embeddings.size(1))
    v_embeddings = {}
    for v, v_list in self._value_id2text.items():
      v_ids = batch_to_id(v_list)
      if 'tokens' in v_ids:
        elmo_embedder_input = {'tokens':v_ids['tokens']}
      elif 'elmo' in v_ids:
        elmo_embedder_input = {'elmo':v_ids['elmo']}
      v_elmo_embeddings = self._text_field_embedder(elmo_embedder_input, num_wrapping_dims=0).sum(1)
      char_embedder_input = {'token_characters':v_ids['token_characters']}
      v_char_embeddings = self._text_field_char_embedder(char_embedder_input, num_wrapping_dims=0).sum(1)
      v_embeddings[v] = torch.cat((v_elmo_embeddings, v_char_embeddings), dim=1)
      v_masks = util.get_text_field_mask(v_ids, num_wrapping_dims=0).sum(1).float()
      v_embeddings[v] = v_embeddings[v] / v_masks.unsqueeze(1).repeat(1, v_embeddings[v].size(1))
    return ds_embeddings, v_embeddings

  def read_domain_slot_list(self, filename):
    with open(filename) as fp:
      lines = fp.readlines()
    domains = []
    domain_slots = []
    value_file_path = {}
    domain_slots_type = {}
    domain_slots_use_value_list = {}
    ds_masked = {}
    num_ds_use_value = 0
    for line in lines:
      line = line.strip("\n ")
      if line.startswith("#"):
        continue
      if len(line.strip("\n ")) == 0 :
        continue
      line_arr = line.split("\t")
      ds = line_arr[0] + " " + line_arr[1]
      if line_arr[3] == "n":
        domains.append(line_arr[0])
        domain_slots.append(ds)
        value_file_path[ds] = line_arr[4].strip(" \n")
        domain_slots_type[ds] = line_arr[2]
        domain_slots_use_value_list[ds] = True if line_arr[5] == "y" else False
        num_ds_use_value += 1 if line_arr[5] == "y" else 0
        ds_masked[ds] = True if line_arr[6] == "y" else False
    ds_text2id = {}
    for i, s in enumerate(domain_slots):
      ds_text2id[s] = i
    return domains, domain_slots, ds_text2id, value_file_path, domain_slots_type, domain_slots_use_value_list, num_ds_use_value, ds_masked

  def load_value_list(self, ds_path):
    def read_value_list(ds_path, ds, value_path_list):
      dir_path = os.path.dirname(ds_path)
      filename = dir_path + "/" + value_path_list[ds]
      with open(filename) as fp:
        lines = fp.readlines()
      values = []
      for line_i, line in enumerate(lines):
        if len(line.strip("\n ")) == 0:
          continue
        values.append(line.strip("\n "))
      value2id = {}
      for i, v in enumerate(values):
        value2id[v] = i
      return values, value2id
    value_text2id = {}
    value_id2text = {}
    for ds in self._ds_text2id.keys():
      if not self._ds_use_value_list[ds]: continue
      id2v, v2id  =read_value_list(ds_path, ds, self.value_file_path)
      value_text2id[ds] = v2id
      value_id2text[ds] = id2v
    return value_id2text, value_text2id

  # code from https://github.com/allenai/allennlp/blob/master/allennlp/models/reading_comprehension/bidaf.py
  def get_best_span(self, span_start_logits, span_end_logits):
      # We call the inputs "logits" - they could either be unnormalized logits or normalized log
      # probabilities.  A log_softmax operation is a constant shifting of the entire logit
      # vector, so taking an argmax over either one gives the same result.
      if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
          raise ValueError("Input shapes must be (batch_size, passage_length)")
      batch_size, passage_length = span_start_logits.size()
      device = span_start_logits.device
      # (batch_size, passage_length, passage_length)
      span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
      # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
      # the span ends before it starts.
      span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                            device=device)).log().unsqueeze(0)
      valid_span_log_probs = span_log_probs + span_log_mask

      # Here we take the span matrix and flatten it, then find the best span using argmax.  We
      # can recover the start and end indices from this flattened list using simple modular
      # arithmetic.
      # (batch_size, passage_length * passage_length)
      best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
      span_start_indices = best_spans // passage_length
      span_end_indices = best_spans % passage_length
      return torch.stack([span_start_indices, span_end_indices], dim=-1)

  def ds_graph_embeddings(self, batch_size, predictions, ds_embeddings, v_embeddings):
    repeated_ds_embeddings = ds_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
    for node_i, node in enumerate(self._ds_id2text):
      if not self._ds_use_value_list[node]: continue
      val_node = v_embeddings[node][predictions[0][node]]
      ds_node = repeated_ds_embeddings[:, self._ds_text2id[node], :]
      ds_node = ds_node + val_node
      repeated_ds_embeddings = repeated_ds_embeddings.clone()
      repeated_ds_embeddings[:, self._ds_text2id[node], :] = ds_node
    return repeated_ds_embeddings

