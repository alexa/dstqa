// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
// Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import os
import pdb
import json
import logging
import numpy as np
from overrides import overrides
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField, SequenceLabelField, ArrayField

from . import dstqa_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("dstqa")
class DSTQAReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 domain_slot_list_path: str = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._sys_user_symbol_indexers = {'symbols': SingleIdTokenIndexer()}
        self._ds_list, self._ds_text2id, value_path_list, self._ds_type, self._ds_use_value_list, self._ds_masked = self.read_domain_slot_list(domain_slot_list_path)
        self._ds_span_list, self._ds_span_text2id = self.ds_span_dict(self._ds_list, self._ds_type)
        self._value_id2text = {}
        self._value_text2id = {}
        for domain_slot in self._ds_list:
          if not self._ds_use_value_list[domain_slot]: continue
          self._value_id2text[domain_slot], self._value_text2id[domain_slot] = self.read_value_list(domain_slot_list_path, domain_slot, value_path_list)

    def ds_span_dict(self, ds_list, ds_type):
      ds_span_list = []
      ds_span_text2id = {}
      for ds in ds_list:
        if ds_type[ds] == "span":
          ds_span_list.append(ds)
          ds_span_text2id[ds] = len(ds_span_list) - 1
      return ds_span_list, ds_span_text2id

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
        logger.info("Reading the dataset")
        for dialog in dataset_json:
            dialog_idx = dialog["dialogue_idx"]
            dialog = dialog['dialogue']
            labels = []
            spans = []
            span_labels = []
            utt_lens = []
            dialog_context = None
            exact_match_feas = None
            tags = None
            for turn_i, turn in enumerate(dialog):
                sys_utt = turn['system_transcript']
                user_utt = turn['transcript']
                tokenized_sys_utt = self._tokenizer.tokenize(sys_utt)
                if turn_i != 0:
                  tokenized_sys_utt = [Token(text="<S>", lemma_="<S>")] + tokenized_sys_utt
                sys_exact_match_feas = self.exact_match(tokenized_sys_utt) 
                sys_tags = [Token("SYS") for token in tokenized_sys_utt]
                tokenized_user_utt = self._tokenizer.tokenize(user_utt)
                if turn_i != len(dialog) - 1:
                  tokenized_user_utt = tokenized_user_utt + [Token(text="</S>", lemma_="</S>")]
                user_exact_match_feas = self.exact_match(tokenized_user_utt)
                user_tags = [Token("USER") for token in tokenized_user_utt]
                utt_lens.append(len(tokenized_sys_utt) + len(tokenized_user_utt))

                if dialog_context is None:
                  dialog_context = tokenized_sys_utt + tokenized_user_utt
                  exact_match_feas = sys_exact_match_feas + user_exact_match_feas
                  tags = sys_tags + user_tags
                else:
                  dialog_context += tokenized_sys_utt + tokenized_user_utt
                  exact_match_feas += sys_exact_match_feas + user_exact_match_feas
                  tags += sys_tags + user_tags
                cur_labels = []
                cur_spans = []
                cur_span_labels = [] # 0: none; 1: dont care; 2: span
                turn_label = turn['belief_state']
                for domain, val in turn_label.items():
                  domain = domain.lower().strip(" \n")
                  val = val["semi"]
                  for slot, value in val.items():
                    ds = domain + " " + slot
                    if ds not in self._ds_text2id: continue
                    slot, value = slot.lower().strip(" \n"), value.lower().strip(" \n")
                    cur_labels.append((domain, slot, value))
                    if ds in self._ds_type and self._ds_type[ds] == "span":
                      s, e = dstqa_util.find_span(dialog_context, value)
                      cur_spans.append((domain, slot, s, e))
                      if value == "dont care": sl = 1
                      elif value == "" or value == "none": sl = 0
                      else: sl = 2
                      cur_span_labels.append((domain, slot, sl))
                labels.append(cur_labels)
                spans.append(cur_spans)
                span_labels.append(cur_span_labels)
          
            instance = self.text_to_instance(dialog_idx, dialog_context, exact_match_feas, tags, utt_lens, labels, spans, span_labels)
            yield instance

    @overrides
    def text_to_instance(self, dialog_idx,
                         dialog_context, exact_match_feas, tags, 
                         utt_lens, labels, spans, span_labels):
        token_indexers = self._token_indexers
        symbol_indexers = self._sys_user_symbol_indexers
        fields: Dict[str, Field] = {}
        fields['dialogs'] = TextField(dialog_context, token_indexers)
        fields['tags'] = TextField(tags, symbol_indexers)
        fields['utt_lens'] = ArrayField(np.array(utt_lens), dtype=np.int32)
        fields['exact_match'] = ListField(exact_match_feas)
        fields['metadata'] = MetadataField(dialog_context)
        fields['dialog_indices'] = MetadataField(dialog_idx)

        # calculate labels
        if labels != None:
          expanded_value_labels = []
          for turn_label in labels:
            turn_value_label = [-1 if self._ds_type[ds] == "span" else 0 for ds in self._ds_list] # 0 is default which is 'none' is in vocab
            for each_label in turn_label:
              if each_label[2] == "":
                continue
              ds = each_label[0] + " " + each_label[1]
              if ds in self._ds_text2id:
                if self._ds_type[ds] == "classification":
                  if each_label[2] not in self._value_text2id[ds]:
                    #print(ds, each_label[2])
                    continue
                  turn_value_label[self._ds_text2id[ds]] = self._value_text2id[ds][each_label[2]]
                if self._ds_type[ds] == "span" and self._ds_use_value_list[ds] == True:
                  if each_label[2] != "none" and each_label[2] != "dont care":
                    if each_label[2] not in self._value_text2id[ds]:
                      #print(ds, each_label[2])
                      continue
                    turn_value_label[self._ds_text2id[ds]] = self._value_text2id[ds][each_label[2]]
            expanded_value_labels.append(ListField([LabelField(l, skip_indexing=True) for l in turn_value_label]))
          fields['labels'] = ListField(expanded_value_labels)

          # calculate spans
          if len(self._ds_span_list) != 0:
            spans_start = []
            spans_end = []
            for turn_span in spans:
              cur_span_start = [-1] * len(self._ds_span_list)
              cur_span_end = [-1] * len(self._ds_span_list)
              for each_span in turn_span:
                cur_ds = each_span[0] + " " + each_span[1]
                cur_span_start[self._ds_span_text2id[cur_ds]] = each_span[2]
                cur_span_end[self._ds_span_text2id[cur_ds]] = each_span[3]
              spans_start.append(ListField([LabelField(l, skip_indexing=True) for l in cur_span_start]))
              spans_end.append(ListField([LabelField(l, skip_indexing=True) for l in cur_span_end]))
            fields["spans_start"] = ListField(spans_start)
            fields["spans_end"] = ListField(spans_end)

            expanded_span_labels = []
            for turn_span_label in span_labels:
              cur_span_label = [0 for _ in self._ds_span_list]
              for each_span_label in turn_span_label:
                cur_ds = each_span_label[0] + " " + each_span_label[1]
                cur_span_label[self._ds_span_text2id[cur_ds]] = each_span_label[2]
              expanded_span_labels.append(ListField([LabelField(l, skip_indexing=True) for l in cur_span_label]))
            fields["span_labels"] = ListField(expanded_span_labels)
        return Instance(fields)

    def read_domain_slot_list(self, filename):
      with open(filename) as fp:
        lines = fp.readlines()
      domain_slots = []
      ds_masked = {}
      value_file_path = {}
      domain_slots_type = {}
      domain_slots_use_value_list = {}
      for line in lines:
        line = line.strip("\n ")
        if line.startswith("#"):
          continue
        if len(line.strip("\n ")) == 0 :
          continue
        line_arr = line.split("\t")
        ds = line_arr[0] + " " + line_arr[1]
        if line_arr[3] == "n":
          domain_slots.append(ds)
          value_file_path[ds] = line_arr[4].strip(" \n")
          domain_slots_type[ds] = line_arr[2]
          domain_slots_use_value_list[ds] = True if line_arr[5] == "y" else False
          ds_masked[ds] = True if line_arr[6] == "y" else False
      ds_text2id = {}
      for i, s in enumerate(domain_slots):
        ds_text2id[s] = i
      return domain_slots, ds_text2id, value_file_path, domain_slots_type, domain_slots_use_value_list, ds_masked

    def read_value_list(self, ds_path, ds, value_path_list):
      dir_path = os.path.dirname(ds_path)
      filename = dir_path + "/" + value_path_list[ds]
      with open(filename) as fp:
        lines = fp.readlines()
      values = []
      for line_i, line in enumerate(lines):
        if self._ds_type[ds] == "span" and line_i < 2: continue # if span, do not read none and dont care
        if len(line.strip("\n ")) == 0:
          continue
        values.append(line.strip("\n "))
      text2id = {}
      for i, v in enumerate(values):
        text2id[v] = i
      return values, text2id

    def exact_match(self, utt):
      def charpos2wordpos(p, utt):
        if p == -1: return p
        num_blank = 0
        for i in range(p):
          if utt[i] == " ": num_blank += 1
        return num_blank
      def num_words(value):
        num_blank = 1
        for i in range(len(value)):
          if value[i] == " ": num_blank += 1
        return num_blank
      # training and test data have already converted to lower cased.
      # keep cases-sensitive should be better
      word_text = " ".join([word.text for word in utt])
      word_lemma_text = " ".join([word.lemma_ for word in utt])
      ds_fea1, ds_fea2 = [], []
      for ds in self._ds_list:
        fea1 = [0] * len(utt)
        fea2 = [0] * len(utt)
        if not self._ds_use_value_list[ds]:
          continue
        for value in self._value_id2text[ds]:
          v_nwords = num_words(value)
          p1 = charpos2wordpos(word_text.find(value), word_text)
          p2 = charpos2wordpos(word_lemma_text.find(value), word_lemma_text)
          if p1 != -1:
            for i in range(p1, p1 + v_nwords):
              fea1[i] = 1
          if p2 != -1:
            for i in range(p2, p2 + v_nwords):
              fea2[i] = 1
        ds_fea1.append(fea1)
        ds_fea2.append(fea2)

      len_utt = len(utt)
      final_output = [[] for _ in range(len_utt)]
      for ori, lemma in zip(ds_fea1, ds_fea2):
        for i, (s_ori, s_lemma) in enumerate(zip(ori, lemma)):
          final_output[i] += [s_ori, s_lemma]
      for i in range(len_utt):
        final_output[i] = ArrayField(np.array(final_output[i]))
      return final_output

