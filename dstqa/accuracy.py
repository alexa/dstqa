// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
// Licensed under the Amazon Software License  http://aws.amazon.com/asl/

from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy
from . import dstqa_util

class Accuracy:
  def __init__(self, ds_id2text, ds_type):
    self._ds_id2text = ds_id2text
    self._ds_type = ds_type
    self._span_text2id, self._class_text2id, self._text2id = dstqa_util.gen_text2id(ds_id2text, ds_type)

    num_span_slot = len(self._span_text2id)
    num_class_slot = len(self._class_text2id)

    self._span_label_acc = [CategoricalAccuracy() for _ in range(num_span_slot)]
    self._span_start_acc = [CategoricalAccuracy() for _ in range(num_span_slot)]
    self._span_end_acc = [CategoricalAccuracy() for _ in range(num_span_slot)]
    self._class_acc = [CategoricalAccuracy() for _ in range(num_class_slot+num_span_slot)]

  def span_label_acc(self, slot_name, logits, labels, label_masks):
    idx = self._span_text2id[slot_name]
    self._span_label_acc[idx](logits, labels, label_masks)
  
  def value_acc(self, slot_name, logits, labels, label_masks):
    idx = self._text2id[slot_name]
    self._class_acc[idx](logits, labels, label_masks)
 
  def span_start_acc(self, slot_name, logits, labels, label_masks):
    idx = self._span_text2id[slot_name]
    self._span_start_acc[idx](logits, labels, label_masks)
   
  def span_end_acc(self, slot_name, logits, labels, label_masks):
    idx = self._span_text2id[slot_name]
    self._span_end_acc[idx](logits, labels, label_masks)

  def get_metrics(self, reset = False):
    acc = {}
    for val_i, val_acc in enumerate(self._class_acc):
      acc["val_" + str(val_i) + "_acc"] = val_acc.get_metric(reset)
    for val_i, val_acc in enumerate(self._span_label_acc):
      acc["sl_" + str(val_i) + "_acc"] = val_acc.get_metric(reset)
    for val_i, val_acc in enumerate(self._span_start_acc):
      acc["ss_" + str(val_i) + "_acc"] = val_acc.get_metric(reset)
    for val_i, val_acc in enumerate(self._span_end_acc):
      acc["se_" + str(val_i) + "_acc"] = val_acc.get_metric(reset)
    return acc

