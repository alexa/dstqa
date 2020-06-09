// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
// Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import json
import os
import numpy as np
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.models import Model
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField, SequenceLabelField, ArrayField

from . import dstqa_util


@Predictor.register('dstqa')
class DSTQAPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        domain_slot_list_path = "./ontology/domain_slot_list_nosp.txt" # if span, use "./ontology/domain_slot_list.txt"
        self._splitter = SpacyWordSplitter(language=language, pos_tags=True, ner=True)
        self._tokenizer = WordTokenizer(word_splitter=self._splitter)
        self._ds_list, self._ds_text2id, value_path_list, self._ds_type, self._ds_use_value_list = self.read_domain_slot_list(domain_slot_list_path)
        self._value_id2text = {}
        self._value_text2id = {}
        for domain_slot in self._ds_list:
          if not self._ds_use_value_list[domain_slot]: continue
          self._value_id2text[domain_slot], self._value_text2id[domain_slot] = self.read_value_list(domain_slot_list_path, domain_slot, value_path_list)

    def predict(self, jsonline: str) -> JsonDict:
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        dialog = json_dict['dialogue']
        dialog_idx = json_dict["dialogue_idx"]

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
        
        instance = self._dataset_reader.text_to_instance(dialog_idx, dialog_context, exact_match_feas, tags, utt_lens, labels, spans, span_labels)
        return instance

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

    def read_domain_slot_list(self, filename):
      with open(filename) as fp:
        lines = fp.readlines()
      domain_slots = []
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
      ds_text2id = {}
      for i, s in enumerate(domain_slots):
        ds_text2id[s] = i
      return domain_slots, ds_text2id, value_file_path, domain_slots_type, domain_slots_use_value_list

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
