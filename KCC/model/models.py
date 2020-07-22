from transformers import AutoModelWithLMHead

import torch
import torch.nn as nn
import torch.nn.functional as F

class KcBERT_FineTuner(nn.Module):
    def __init__(self, intent_class_num, entity_class_num, default_model_path='beomi/kcbert-base'):
        super(KcBERT_FineTuner, self).__init__()

        self.backbone = AutoModelWithLMHead.from_pretrained(default_model_path)
        self.feature_dim = self.backbone.config.vocab_size #not hidden size

        self.intent_class_num = intent_class_num
        self.entity_class_num = entity_class_num
        self.intent_embedding = nn.Linear(self.feature_dim, self.intent_class_num)
        self.entity_embedding = nn.Linear(self.feature_dim, self.entity_class_num)

        self.pad_token_id = self.backbone.config.pad_token_id
        self.max_seq_len = self.backbone.config.max_position_embeddings

    def forward(self, tokens):
        attention_mask = (tokens != self.pad_token_id).type_as(tokens).float()
        position_ids = torch.arange(self.max_seq_len).repeat(tokens.size(0), 1).type_as(tokens)

        feature = self.backbone(tokens, attention_mask=attention_mask, position_ids=position_ids)[0]

        intent_pred = self.intent_embedding(feature[:,0,:]) #forward only first [CLS] token
        entity_pred = self.entity_embedding(feature[:,1:,:]) #except first [CLS] token


        return intent_pred, entity_pred

        

        
        




