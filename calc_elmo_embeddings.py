// Copyright <first-edit-year> Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
// Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# pre-calculate elmo embeddings of each sentence in each dialog
import sys
import pdb
import json
import pickle
from tqdm import tqdm

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

base_path = sys.argv[1]
train_data_path = base_path + "/train.json"
dev_data_path = base_path + "/dev.json"
test_data_path = base_path + "/test.json"
data_paths = {"train": train_data_path, "dev": dev_data_path, "test": test_data_path}
data_path = data_paths[sys.argv[2]]
output_path = sys.argv[3] + "/" + sys.argv[2]
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"


def read_dataset(file_path):
  with open(file_path) as dataset_file:
    tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
    dataset_json = json.load(dataset_file)
    dialogs = []
    for dialog in dataset_json:
      dialog_idx = dialog["dialogue_idx"]
      dialog = dialog['dialogue']
      dialog_context = None
      for turn_i, turn in enumerate(dialog):
        sys_utt = turn['system_transcript']
        user_utt = turn['transcript']
        tokenized_sys_utt = tokenizer.tokenize(sys_utt)
        if turn_i != 0:
          tokenized_sys_utt = [Token(text="<S>", lemma_="<S>")] + tokenized_sys_utt
        tokenized_user_utt = tokenizer.tokenize(user_utt)
        if turn_i != len(dialog) - 1:
          tokenized_user_utt = tokenized_user_utt + [Token(text="</S>", lemma_="</S>")]
        if dialog_context is None:
          dialog_context = tokenized_sys_utt + tokenized_user_utt
        else:
          dialog_context += tokenized_sys_utt + tokenized_user_utt
      dialog_context = [t.text for t in dialog_context]
      dialogs.append((dialog_idx, [dialog_context]))
  return dialogs

def calc_elmo_embeddings(elmo, dialog):
  # Compute two different representation for each token.
  # Each representation is a linear weighted combination for the
  # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
  
  # use batch_to_ids to convert sentences to character ids
  character_ids = batch_to_ids(dialog).cuda()
  dialog_embeddings = []
  for i in range(3):
    embeddings = elmo[i](character_ids)
    batch_embeddings = embeddings['elmo_representations'][0]
    batch_embeddings = batch_embeddings.squeeze(0)
    dialog_embeddings.append(batch_embeddings.cpu())
 
  return dialog_embeddings 
 

#https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md 
#After loading the pre-trained model, the first few batches will be negatively impacted until the biLM can reset its internal states. You may want to run a few batches through the model to warm up the states before making predictions (although we have not worried about this issue in practice).
def elmo_warm_up(elmo, dialog):
  character_ids = batch_to_ids(dialog).cuda()
  for i in range(3):
    for _ in range(20):
      elmo[i](character_ids)
  
elmo = [None] * 3 
elmo[0] = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[1.0, 0, 0]).cuda()
elmo[1] = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[0, 1.0, 0]).cuda()
elmo[2] = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[0, 0, 1.0]).cuda()
dialogs = read_dataset(data_path)
elmo_warm_up(elmo, dialogs[0][1])
dialog_embeddings = {}
for dialog_idx, dialog in tqdm(dialogs):
  dialog_embedding = calc_elmo_embeddings(elmo, dialog)
  dialog_embeddings[dialog_idx] = dialog_embedding

with open(output_path, 'wb') as handle:
  pickle.dump(dialog_embeddings, handle)

