import torch
import torch.nn as nn


def compute_energy(logits, T=0.5):
    """Compute energy score with temperature scaling."""
    return -T * torch.logsumexp(logits / T, dim=1)


class Model(nn.Module):   
    def __init__(self, decoder, tokenizer, args, num_labels):
        super(Model, self).__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.args = args
        self.score = nn.Linear(decoder.config.n_embd, 2, bias=False)
    
    def forward(self, 
                input_ids, 
                labels=None, 
                return_energy_only=False, 
                return_hidden_state=False):
        
        # Identify the last actual token location (skip padding)
        last_token_loc = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1
        
        # Enable hidden states + return_dict
        decoder_outputs = self.decoder(
            input_ids, 
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            output_hidden_states=True,
            return_dict=True
        )
        
        # "all_hidden_states" is a tuple of shape (num_layers+1) each [batch_size, seq_len, hidden_dim]
        # The final hidden layer is decoder_outputs.hidden_states[-1]
        last_hidden_states = decoder_outputs.hidden_states[-1]
        
        # Compute logits from the final hidden layer
        logits = self.score(last_hidden_states)     # shape [batch_size, seq_len, 2]
        logits = logits[torch.arange(input_ids.shape[0], device=self.args.device), last_token_loc]  # [batch_size, 2]
                
        # Compute energy from the final logits
        energy_score = compute_energy(logits)
        
        # If only energy requested, return early
        if return_energy_only:
            return energy_score
        
        # If we have labels => training or validation step
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)            
            # Return hidden states if requested
            if return_hidden_state:
                # Return the entire hidden-states tuple to access penultimate layers if needed
                return loss, energy_score, decoder_outputs.hidden_states
            else:
                return loss, energy_score
        
        # Else: Inference mode (no labels)
        else:
            prob = torch.softmax(logits, dim=-1)         # shape [batch_size, 2]            
            if return_hidden_state:
                # Return the entire hidden-states tuple
                return prob, energy_score, decoder_outputs.hidden_states
            else:
                return prob, energy_score