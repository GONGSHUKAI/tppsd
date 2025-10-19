import dpp
import torch
import torch.nn as nn
import math
from torch.distributions import Categorical
from .THP import thp_Encoder
from .SAHP import sahp_Encoder
from .ATTNHP import attnhp_Encoder  
from dpp.data.batch import Batch
from dpp.utils import diff
import time
PAD_TOKEN_ID = 0

class TransformerTPP(nn.Module):
    def __init__(
        self,
        num_marks: int,
        encoder_type: str = "thp",
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Create a TPP model that uses transformer to aggregate history information

        Args:
            (Args for TPP model)
            num_marks: Number of marks (i.e. classes / event types)
            mean_log_inter_time: Average log-inter-event-time
            std_log_inter_time: Std of log-inter-event-times
            context_size: Size of the history embedding
            mark_embedding_size: Size of the mark embedding

            (Args for Transformer)
            nhead: Number of heads in the multiheadattention models
            num_layers: Number of TransformerEncoder layers in the encoder
            dropout: Dropout value
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size

        self.mark_linear = nn.Linear(self.context_size, self.num_marks)        
        self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the serial
        self.nhead = nhead
        self.num_layers = num_layers

        ######## Transformer Architecture ########
        assert context_size % nhead == 0, "context_size must be divisible by nhead"
        if self.encoder_type == "thp":
            self.thp_encoder = thp_Encoder(
                num_types=self.num_marks,
                d_model=context_size,
                d_inner=context_size//2,
                n_layers=num_layers,
                n_head=nhead,
                d_k=context_size//nhead,
                d_v=context_size//nhead,
                dropout=dropout,
            )
        elif self.encoder_type == "sahp":
            self.sahp_encoder = sahp_Encoder(
                num_types=self.num_marks,
                d_model=context_size,
                n_head=nhead,
                n_layers=num_layers,
                dropout=dropout,
                pad_token_id=PAD_TOKEN_ID
            )
        elif self.encoder_type == "attnhp":
            self.attnhp_encoder = attnhp_Encoder(
                num_types=self.num_marks,
                d_model=context_size,
                n_head=nhead,
                num_layers=num_layers,
                dropout=dropout,
                pad_token_id=PAD_TOKEN_ID
            )
    def get_context(self, batch: dpp.data.Batch, 
                          remove_last: bool = True) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        event_times = batch.event_times
        if self.num_marks > 1:
            event_types = batch.marks
        else:
            event_types = None
        non_pad_mask = batch.mask.bool()

        if self.encoder_type == "thp":
            context = self.thp_encoder(event_types, event_times, non_pad_mask)

        elif self.encoder_type == 'sahp':
            context = self.sahp_encoder(event_types, event_times, non_pad_mask)
        
        elif self.encoder_type == 'attnhp':
            context = self.attnhp_encoder(event_types, event_times, non_pad_mask)

        batch_size, _, _ = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)

        context = torch.cat([context_init, context], dim=1)
        if remove_last:
            context = context[:, :-1, :]

        return context  # (batch_size, seq_len, context_size)
    
    def log_prob(self, batch: dpp.data.Batch) -> torch.Tensor:
        """
        Compute log-likelihood for a batch of sequences.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            log_p: shape (batch_size,), each element is the log-likelihood of one sequence in the batch.
        """
        context = self.get_context(batch)  # (batch_size, seq_len, context_size)
        inter_time_dist = self.get_inter_time_dist(context) # A LogNormalMixtureDistribution object
        inter_times = batch.inter_times.clamp(1e-10)        # avoid log(0)
        log_p = inter_time_dist.log_prob(inter_times)

        # Compute the survival log-likelihood from t_N to t_end
        # A sample mask: [[1,1,1,0], [1,1,0,0]]，therefore mask.sum(-1, keepdim=True) is [[3],[2]], indicating the index of the last event in each sequence
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len), compute the survival likelihood from t_1 to t_N
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,), we need the survival likelihood at the last event only

        del inter_time_dist, inter_times, log_surv_all, last_event_idx

        if self.num_marks > 1:  # compute the log-likelihood of marks
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # MLP layer for classification, (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
            del mark_logits, mark_dist
        log_p *= batch.mask  # (batch_size, seq_len)
        # torch.cuda.empty_cache()
        return (log_p.sum(-1) + log_surv_last)/batch.t_ends  # (batch_size,)

    def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            context_init = self.context_init
        else:
            print(context_init.shape)
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0)
        if self.num_marks > 1:
            marks = torch.empty(batch_size, 0, dtype=torch.long)

        generated = False
        start_time = time.time()
        count = 0
        while not generated:
            sample_one_time = time.time()
            inter_time_dist = self.get_inter_time_dist(next_context)
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)

            # Generate marks, if necessary
            if self.num_marks > 1:
                mark_logits = torch.log_softmax(self.mark_linear(next_context), dim=-1)  # (batch_size, 1, num_marks)
                mark_dist = Categorical(logits=mark_logits)
                next_marks = mark_dist.sample()  # (batch_size, 1)
                marks = torch.cat([marks, next_marks], dim=1)
            else:
                marks = None

            with torch.no_grad():
                generated = inter_times.sum(-1).min() >= t_end
                print(inter_times.sum(-1).min())

            count += 1
            print(f'Finished sampling timestep {count} in {time.time() - sample_one_time:.2f} seconds')
            print(f'inter_times: {inter_times.shape}')
            print(f'event_times: {inter_times.cumsum(-1).shape}')
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), event_times=inter_times.cumsum(-1),
                          marks=marks, t_ends=torch.full((batch_size,), t_end))
            context = self.get_context(batch, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)

        print(f"Sampling took {time.time() - start_time:.2f} seconds")
        arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.num_marks > 1:
            marks = marks * mask  # (batch_size, seq_len)
        return Batch(inter_times=inter_times, mask=mask, event_times=arrival_times, marks=marks, t_ends=torch.full((batch_size,), t_end))
    