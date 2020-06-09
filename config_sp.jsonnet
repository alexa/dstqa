local train_data_path = "./data/train.json";
local validation_data_path = "./data/dev.json";
local elmo_embedding_path = "./data/elmo_embeddings/all/";
local domain_slot_list_path = "./ontology/domain_slot_list_sp.txt";
local base_dim = 612; # 512 + 100
local symbol_dim = 128;
local num_use_val = 25; # number of values that uses value list
local dropout = 0.5;
local bi_dropout = 0.5;
local save_mem = false;
local vocab_min_count = 0;
local loss_scale_by_num_values = false;
local is_embedding_trainable = false;
local use_pre_calc_elmo_embeddings = true;
local use_graph = true;
local word_embeddings = "elmo"; # glove or elmo
local phrase_layer_type = "gru"; # gru or stacked_self_attention
{
    "dataset_reader": {
        "type": "dstqa",
        "lazy": false,
        "tokenizer": {
          "type": "word",
          "word_splitter": {
            "type": "spacy",
            "pos_tags": false,
            "ner": false,
          },
        }, 
        "token_indexers": if use_pre_calc_elmo_embeddings == false then {
            [if word_embeddings == "glove" then "tokens" else "elmo"]: 
            if word_embeddings == "glove" then {"type": "single_id"}
            else {"type": "elmo_characters"},
            "token_characters": {
                "type": "characters",
                "character_tokenizer": {
                    "byte_encoding": "utf-8",
                    "end_tokens": [
                        260
                    ],
                    "start_tokens": [
                        259
                    ]
                },
                "min_padding_length": 5
            }
        } else {
          "token_characters": {
              "type": "characters",
              "character_tokenizer": {
                  "byte_encoding": "utf-8",
                  "end_tokens": [
                      260
                  ],
                  "start_tokens": [
                      259
                  ]
              },
              "min_padding_length": 5
          }
        },
        "domain_slot_list_path": domain_slot_list_path,
    },
    "iterator": {
        "track_epoch": true,
        "type": "bucket",
        "batch_size": 36,
        "max_instances_in_memory": 1000,
        "sorting_keys": [
            [
                "dialogs",
                "num_tokens"
            ]
        ]
    },
    "vocabulary": {
        "min_count": {
            "tokens": vocab_min_count
        },
    },
    "model": {
        "use_graph": use_graph,
        "loss_scale_by_num_values": loss_scale_by_num_values,
        "use_pre_calc_elmo_embeddings": use_pre_calc_elmo_embeddings,
        "elmo_embedding_path": elmo_embedding_path,
        "base_dim": base_dim,
        "domain_slot_list_path": domain_slot_list_path,
        "type": "dstqa",
        "dropout": dropout,
        "bi_dropout": bi_dropout,
        "word_embeddings": word_embeddings,
        "initializer": [],
        "token_indexers": {
            [if word_embeddings == "glove" then "tokens" else "elmo"]: 
            if word_embeddings == "glove" then {"type": "single_id"}
            else {"type": "elmo_characters"},
            "token_characters": {
                "type": "characters",
                "character_tokenizer": {
                    "byte_encoding": "utf-8",
                    "end_tokens": [
                        260
                    ],
                    "start_tokens": [
                        259
                    ]
                },
                "min_padding_length": 5
            }
        },
        "text_field_char_embedder": {
            "token_characters": {
                "type": "character_encoding",
                "dropout": 0.0,
                "embedding": {
                    "embedding_dim": 20,
                    "num_embeddings": 262
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 20,
                    "ngram_filter_sizes": [
                        5
                    ],
                    "num_filters": 100
                }
            }
        },
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            [if word_embeddings == "glove" then "tokens" else "elmo"]:
            if word_embeddings == "glove" then {
              "type": "embedding",
              "embedding_dim": 300,
              "pretrained_file": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
              "trainable": is_embedding_trainable
            }
            else {
                "type": "elmo_token_embedder",
                "do_layer_norm": false,
                "requires_grad": is_embedding_trainable,
                "dropout": 0.0,
                "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json",
                "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
            },
        },
        "symbol_embedder": {
          "symbols": {
            "type": "embedding",
            "num_embeddings": 40,
            "embedding_dim": symbol_dim
          }
        },
        "phrase_layer": 
          if phrase_layer_type == "gru" then {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": base_dim / 2,
            "input_size": base_dim + symbol_dim + 2 * num_use_val,
            "num_layers": 1
          } else {
            "type": "stacked_self_attention",
            "hidden_dim": base_dim / 2,
            "projection_dim": base_dim / 2,
            "feedforward_hidden_dim": base_dim / 2,
            "num_attention_heads": 4,
            "input_dim": base_dim + symbol_dim + 2 * num_use_val,
            "num_layers": 1,
            "use_positional_encoding": true
          },
        "class_prediction_layer": {
          "input_dim": base_dim,
          "num_layers": 1,
          "hidden_dims": [base_dim],
          "activations": ["linear"],
        },
        "span_prediction_layer": {
          "input_dim": base_dim,
          "num_layers": 1,
          "hidden_dims": [base_dim],
          "activations": ["linear"],
        },
        "span_label_predictor": {
          "input_dim": base_dim,
          "num_layers": 2,
          "hidden_dims": [base_dim, 3],
          "activations": ["relu", "linear"],
        },
        "span_end_encoder": {
          "input_dim": base_dim,
          "num_layers": 1,
          "hidden_dims": base_dim,
          "activations": "relu",
        },
        "span_start_encoder": {
          "input_dim": base_dim,
          "num_layers": 1,
          "hidden_dims": base_dim,
          "activations": "relu",
        },
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "trainer": {
        "num_serialized_models_to_keep": 1,
        "keep_serialized_model_every_num_seconds": 60 * 30,
        "cuda_device": 0,
        "num_epochs": 1000,
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
        }
    },
    "validation_iterator": {
        "track_epoch": true,
        "type": "bucket",
        "batch_size": 36,
        "max_instances_in_memory": 1000,
        "sorting_keys": [
            [
                "dialogs",
                "num_tokens"
            ]
        ]
    }
}

