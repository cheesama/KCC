from .kcc_lightning_model import KcBERT_Classifier

import torch
import torch.nn as nn

import logging

model = None
intent_dict = {}
entity_dict = {}

class Inferencer:
    def __init__(self, checkpoint_path: str):
        self.model = ElectrasaClassifier.load_from_checkpoint(checkpoint_path)
        self.model.model.eval()

        self.intent_dict = {}
        for k, v in self.model.dataset.intent_dict.items():
            self.intent_dict[v] = k # str key -> int key

        self.entity_dict = {}
        for k, v in self.model.dataset.entity_dict.items():
            self.entity_dict[v] = k # str key -> int key

        logging.info("intent dictionary")
        logging.info(self.intent_dict)
        print()

        logging.info("entity dictionary")
        logging.info(self.entity_dict)

    def is_same_entity(self, i, j):
        # check whether XXX_B, XXX_I tag are same 
        return self.entity_dict[i][:self.entity_dict[i].rfind('_')].strip() == self.entity_dict[j][:self.entity_dict[j].rfind('_')].strip()

    def inference(self, text: str, intent_topk=5):
        if self.model is None:
            raise ValueError(
                "model is not loaded, first call load_model(checkpoint_path)"
            )

        text = text.lower()

        # encode text to token_indices
        tokens = self.model.dataset.encode(text)
        intent_result, entity_result = self.model.forward(tokens.unsqueeze(0))

        # mapping intent result
        rank_values, rank_indicies = torch.topk(
            nn.Softmax(dim=1)(intent_result)[0], k=intent_topk
        )
        intent = {}
        intent_ranking = []
        for i, (value, index) in enumerate(
            list(zip(rank_values.tolist(), rank_indicies.tolist()))
        ):
            intent_ranking.append(
                {"confidence": value, "name": self.intent_dict[index]}
            )

            if i == 0:
                intent["name"] = self.intent_dict[index]
                intent["confidence"] = value

        # mapping entity result
        entities = []

        # except first & last sequnce token whcih indicate EOS or [SEP] token
        # SOS or [CLS] token already excepted in model forward step
        entity_indices = (entity_result)[0][:-1]
        start_idx = -1

        #print (self.model.dataset.tokenize(text))

        #print ('tokens')
        #print (tokens)
        #print ('predicted entities')
        #print (entity_indices)

        start_token_position = -1

        # except first sequnce token whcih indicate BOS or [CLS] token
        if type(tokens) == torch.Tensor:
            tokens = tokens.long().tolist()

        for i, entity_idx_value in enumerate(entity_indices):
            if entity_idx_value != 0 and start_token_position == -1:
                start_token_position = i
            elif start_token_position >= 0 and not self.is_same_entity(entity_indices[i-1],entity_indices[i]):
                end_token_position = i - 1

                #print ('start_token_position')
                #print (start_token_position)
                #print ('end_token_position')
                #print (end_token_position)

                # find start text position
                token_idx = tokens[start_token_position + 1]
                token_value = self.model.dataset.tokenizer.convert_ids_to_tokens([token_idx])[0].replace("#", "")

                if len(token_value.strip()) == 0:
                    start_token_position = -1
                    continue
                
                start_position = text.find(token_value.strip())

                # find end text position
                token_idx = tokens[end_token_position + 1]
                token_value = self.model.dataset.tokenizer.convert_ids_to_tokens([token_idx])[0].replace("#", "")

                end_position = text.find(token_value.strip(), start_position) + len(token_value.strip())

                if self.entity_dict[entity_indices[i-1]] != "O": # ignore 'O' tag
                    entities.append(
                        {
                            "start": start_position,
                            "end": end_position,
                            "value": text[start_position:end_position],
                            "entity": self.entity_dict[entity_indices[i-1]][:self.entity_dict[entity_indices[i-1]].rfind('_')]
                        }
                    )
                    
                    start_token_position = -1

                if entity_idx_value == 0:
                    start_token_position = -1

        result = {
            "text": text,
            "intent": intent,
            "intent_ranking": intent_ranking,
            "entities": entities,
        }

        # print (result)

        return result

        # rasa NLU entire result format
        """
        {
            "text": "Hello!",
            "intent": {
                "confidence": 0.6323,
                "name": "greet"
            },
            "intent_ranking": [
                {
                    "confidence": 0.6323,
                    "name": "greet"
                }
            ],
            "entities": [
                {
                    "start": 0,
                    "end": 0,
                    "value": "string",
                    "entity": "string"
                }
            ]
        }
        """
