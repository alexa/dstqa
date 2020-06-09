// Copyright <first-edit-year> Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
// Licensed under the Amazon Software License  http://aws.amazon.com/asl/

def find_span(utt, ans):
  def is_match(utt, ans, i):
    match = True
    for j in range(len(ans)):
      if utt[i+j].text.lower() != ans[j]:
        match = False
    return match

  ans = ans.lower()
  ans = ans.split(" ")
  # find ans from revert direction
  ans_len = len(ans)
  utt_len = len(utt)
  span_start = -1
  span_end = -1
  for i in range(utt_len - ans_len - 1, -1, -1):
    if is_match(utt, ans, i):
      span_start = i
      span_end = span_start + ans_len - 1
      break
  return span_start, span_end

def gen_id2text(ds_id2text, ds_type):
  span_id2text, class_id2text = [], []
  for ds in ds_id2text:
    if ds_type[ds] ==  "span":
      span_id2text.append(ds)
    if ds_type[ds] ==  "classification":
      class_id2text.append(ds)
  return span_id2text, class_id2text

def gen_text2id(ds_id2text, ds_type):
  s_i = 0
  c_i = 0
  i = 0
  span_text2id, class_text2id, text2id = {}, {}, {}
  for ds in ds_id2text:
    if ds_type[ds] ==  "span":
      span_text2id[ds] = s_i
      s_i += 1
    if ds_type[ds] ==  "classification":
      class_text2id[ds] = c_i
      c_i += 1
    text2id[ds] = i
    i+=1
  return span_text2id, class_text2id, text2id

