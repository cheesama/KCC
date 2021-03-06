from collections import OrderedDict, Counter
from tqdm import tqdm
from typing import List

from transformers import AutoTokenizer

import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import re

class RasaIntentEntityDataset(torch.utils.data.Dataset):
    """
    RASA NLU markdown file lines based Custom Dataset Class

    Dataset Example in nlu.md

    ## intent:intent_데이터_자동_선물하기_멀티턴                <- intent name
    - T끼리 데이터 주기적으로 보내기                            <- utterance without entity
    - 인터넷 데이터 [달마다](Every_Month)마다 보내줄 수 있어?    <- utterance with entity
    
    """

    def __init__(self, markdown_lines: List[str], default_model_path='beomi/kcbert-base'):
        self.intent_dict = {}
        self.entity_dict = {}
        self.entity_dict["O"] = 0  # using BIO tagging

        self.dataset = []

        intent_value_list = []
        entity_type_list = []

        current_intent_focus = ""

        text_list = []

        for line in tqdm(
            markdown_lines,
            desc="Organizing Intent & Entity dictionary in NLU markdown file ...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    intent_value_list.append(line.split(":")[1].strip())
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""

            else:
                if current_intent_focus != "":
                    text = line[2:].strip().lower()

                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

                    text = re.sub(r"\([a-zA-Z_1-2]+\)", "", text)  # remove (...) str
                    text = text.replace("[", "").replace(
                        "]", ""
                    )  # remove '[',']' special char

                    if len(text) > 0:
                        text_list.append(text.strip())

        #dataset tokenizer setting
        self.tokenizer = AutoTokenizer.from_pretrained(default_model_path)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id 
        self.eos_token_id = self.tokenizer.sep_token_id #[SEP] token
        self.bos_token_id = self.tokenizer.cls_token_id #[CLS] token

        intent_value_list = sorted(intent_value_list)
        for intent_value in intent_value_list:
            if intent_value not in self.intent_dict.keys():
                self.intent_dict[intent_value] = len(self.intent_dict)

        entity_type_list = sorted(entity_type_list)
        for entity_type in entity_type_list:
            if entity_type + '_B' not in self.entity_dict.keys():
                self.entity_dict[str(entity_type) + '_B'] = len(self.entity_dict)
            if entity_type + '_I' not in self.entity_dict.keys():
                self.entity_dict[str(entity_type) + '_I'] = len(self.entity_dict)

        current_intent_focus = ""

        for line in tqdm(
            markdown_lines, desc="Extracting Intent & Entity in NLU markdown files...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""
            else:
                if current_intent_focus != "":  # intent & entity sentence occur case
                    text = line[2:].strip().lower()

                    entity_value_list = []
                    for value in re.finditer(r"\[(.*?)\]", text):
                        entity_value_list.append(
                            text[value.start() + 1 : value.end() - 1]
                            .replace("[", "")
                            .replace("]", "")
                        )

                    entity_type_list = []
                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

                    text = re.sub(r"\([a-zA-Z_1-2]+\)", "", text)  # remove (...) str
                    text = text.replace("[", "").replace(
                        "]", ""
                    )  # remove '[',']' special char

                    if len(text) > 0:
                        each_data_dict = {}
                        each_data_dict["text"] = text.strip()
                        each_data_dict["intent"] = current_intent_focus
                        each_data_dict["intent_idx"] = self.intent_dict[
                            current_intent_focus
                        ]
                        each_data_dict["entities"] = []

                        for value, type_str in zip(entity_value_list, entity_type_list):
                            for entity in re.finditer(value, text):
                                entity_tokens = self.tokenize(value)

                                for i, entity_token in enumerate(entity_tokens):
                                    if i == 0:
                                        BIO_type_str = type_str + '_B'
                                    else:
                                        BIO_type_str = type_str + '_I'

                                    each_data_dict["entities"].append(
                                        {
                                            "start": text.find(entity_token, entity.start(), entity.end()),
                                            "end": text.find(entity_token, entity.start(), entity.end()) + len(entity_token),
                                            "entity": type_str,
                                            "value": entity_token,
                                            "entity_idx": self.entity_dict[BIO_type_str],
                                        }
                                    )


                        self.dataset.append(each_data_dict)
        
        print(f"Intents: {self.intent_dict}")
        print(f"Entities: {self.entity_dict}")

    def tokenize(self, text: str, skip_special_char=True):
        if skip_special_char:
            return self.tokenizer.tokenize(text)
        else:
            return [token.replace('#','') for token in self.tokenizer.tokenize(text)]
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, padding=True):
        tokens = self.tokenize(self.dataset[idx]["text"])
        token_ids = self.tokenizer.encode(self.dataset[idx]["text"])

        intent_idx = self.dataset[idx]['intent_idx']

        entity_idx = [0] * self.tokenizer.max_len # O tag indicate 0(zero)

        for entity_info in self.dataset[idx]["entities"]:
            ##check whether entity value is include in splitted token
            for token_seq, token_value in enumerate(tokens):
                # Consider [CLS], [SEP] token
                if token_seq == 0 or token_seq == len(tokens) - 1:
                    continue

                for entity_seq, entity_info in enumerate(self.dataset[idx]["entities"]):
                    if token_value == entity_info["value"]:
                        entity_idx[token_seq] = entity_info["entity_idx"]
                        break

        if padding:
            token_ids = token_ids + ([self.pad_token_id] * (self.tokenizer.max_len - len(token_ids)))

        token_ids = torch.from_numpy(np.array(token_ids))
        intent_idx = torch.from_numpy(np.array(intent_idx))
        # Ignore first [CLS] token from entity prediction
        entity_idx = torch.from_numpy(np.array(entity_idx[1:]))

        return token_ids, intent_idx, entity_idx

    def get_intent_idx(self):
        return self.intent_dict

    def get_entity_idx(self):
        return self.entity_dict

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

