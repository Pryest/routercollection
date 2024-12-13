import torch
import os
from transformers import AutoModel
from transformers import AutoModelForCausalLM


class CalibrationEncoder(torch.nn.Module):
    def __init__(self, load_from):
        super(CalibrationEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(load_from)
        self.pik = torch.nn.Linear(self.model.config.hidden_size, 2, bias=False, dtype=self.model.dtype)
        self.pik.weight.data.fill_(0)
    
    def forward(self, *args, **kwargs):
        hidden_states = self.model(*args, **kwargs).last_hidden_state
        logits = self.pik(hidden_states)[:, 0]
        return logits
    
    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.pik.state_dict(), os.path.join(path, "pik.pt"))

    def load(self, path):
        self.model = AutoModel.from_pretrained(path)
        self.pik.load_state_dict(torch.load(os.path.join(path, "pik.pt")))
    

class CalibrationDecoder(torch.nn.Module):
    def __init__(self, load_from):
        super(CalibrationDecoder, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(load_from)
        self.pik = torch.nn.Linear(self.model.config.hidden_size, 2, bias=False, dtype=self.model.dtype)
        self.pik.weight.data.fill_(1)
    
    def forward(self, *args, **kwargs):
        hidden_states = self.model.model(*args, **kwargs).last_hidden_state
        logits = self.pik(hidden_states)[:, -1]
        return logits
    
    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.pik.state_dict(), os.path.join(path, "pik.pt"))
    
    def load(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.pik.load_state_dict(torch.load(os.path.join(path, "pik.pt")))
    
