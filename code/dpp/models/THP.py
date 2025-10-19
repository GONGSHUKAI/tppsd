import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Constants import PAD, GRID, GRID_LENGTH


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    # temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_non_event_mask(seq):
    """ For masking out the non-event time point"""
    len_q = seq.size(1)
    padding_mask = seq.eq(GRID)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask





class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x




class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn



class thp_Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))
        # )

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD).to('cuda')
        # self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        tt = time.unsqueeze(-1) / self.position_vec
        mask = torch.zeros_like(tt).bool()
        result = torch.zeros_like(tt)
        mask[..., 0::2] = True

        result += torch.sin(tt)*mask
        result += torch.cos(tt)*~mask
        return result * non_pad_mask.unsqueeze(-1)

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        if event_type is not None:
            # prepare attention masks
            # slf_attn_mask is where we cannot look, i.e., the future and the padding
            slf_attn_mask_subseq = get_subsequent_mask(event_type)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            enc_output = self.event_emb(event_type)
            if event_type.min() < 0:
                print("wrong event type")
            for name, param in self.event_emb.named_parameters():
                if torch.any(torch.isnan(param.data)):
                    print("wrong parameter")
        else:
            slf_attn_mask_subseq = get_subsequent_mask(non_pad_mask)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=non_pad_mask, seq_q=non_pad_mask)
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            enc_output = 0


        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask.unsqueeze(-1),
                slf_attn_mask=slf_attn_mask)
        return enc_output





class thp_Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=16, d_inner=8,
            n_layers=1, n_head=1, d_k=16, d_v=16, dropout=0.1, opt=None):
        super().__init__()

        self.encoder = thp_Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.name = 'thp'
        self.num_types = num_types
        self.normalize = None
        self.d_inner = d_inner
        self.data_name = opt.data_name
        # self.method = opt.method


        self.base_layer = nn.Sequential(
                nn.Linear(d_model, num_types, bias=True)
                )

        self.affect_layer = nn.Sequential(
                nn.Linear(d_model, num_types, bias=True),
                nn.Tanh()
                )
        self.intensity_layer = nn.Sequential(
                nn.Softplus(beta=1.0)
                )
        
        self.inconsistent_T = opt.inconsistent_T
    
    def forward(self, event_type, event_time, time_gap, opt):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        time_gap = torch.cat((event_time[:,0:1], time_gap), axis = 1)  
        event_time = torch.concatenate((torch.zeros(event_time.shape[0], 1).to(opt.device), event_time), axis = 1)
        event_type = torch.cat((torch.ones(event_type.shape[0], 1).type(torch.long).to(opt.device), event_type), axis = 1)

        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)

        self.enc_output = enc_output
        if opt.method == "wsm":
            loss = self.compute_loss_wsm(event_type, event_time, time_gap, non_pad_mask, opt.h_type, opt.CE_coef)
        elif opt.method == "mle":
            loss = self.compute_loss_mle(event_type, event_time, time_gap, non_pad_mask, num_grid = opt.num_grid)
        elif opt.method == "dsm":
            loss = self.compute_loss_dsm(event_type, event_time, time_gap, non_pad_mask, opt.noise_var, opt.CE_coef)
        else:
            raise ValueError("No such method")
        return loss, enc_output


        
    
    def compute_loss_wsm(self, event_type, event_time, time_gap, non_pad_mask, h_type="two_side_op",alpha = 1):
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        t_var = torch.autograd.Variable(diff_time, requires_grad=True)

        all_intensity, score = self.get_intensity_n_score(t_var, event_type, event_time, time_gap, non_pad_mask)
        score_grad = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0]
        sum_intensity  = all_intensity.sum(-1)

        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_time.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_time.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_time.device)

        type_intensity = (all_intensity * type_mask[:, 1:, :]).sum(-1)

        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()

        if self.data_name not in ["exp-decay-multivariate", 'half-sin_multivariate']:
            # max_observed =  torch.max(event_time, axis = 1)[0].unsqueeze(-1)
            if self.inconsistent_T:
                max_observed = max_observed =  torch.max(event_time, axis = 1)[0].unsqueeze(-1)
            else:
                max_observed =  event_time.max()
        elif self.data_name == "exp-decay-multivariate":
            max_observed = 10
        else:
            max_observed = 5
        
        
        if h_type == "two_side_op":
            t_prior = event_time[:,:-1]
            h = (max_observed - t_prior)/2 - abs(event_time[:,1:] - (max_observed + t_prior)/2)
            hprime = torch.where(event_time[:,1:] > (max_observed + t_prior)/2, -1, 1)
        elif h_type == "None":
            h = torch.ones_like(event_time[:,1:])
            hprime = torch.zeros_like(event_time[:,1:])
        elif h_type == "two_side_ord":
            h = (max_observed - event_time[:,1:]) * time_gap
            hprime = max_observed - event_time[:,1:] - time_gap
        elif h_type == "one_side_ord":
            h = time_gap ** 2
            hprime = 2 * time_gap       
        elif h_type == "one_side_opt":
            h = time_gap
            hprime = torch.ones_like(time_gap)
        else:
            raise ValueError("No such h_type")

    
        WSMLoss = (0.5 * h * score ** 2 + score_grad * h + score * hprime)

        loss = ( WSMLoss + alpha * CELoss) * non_pad_mask[:,1:].squeeze(-1)
        return loss
    

    def compute_loss_dsm(self, event_type, event_time, time_gap, non_pad_mask, var_noise = 0.5, alpha = 1, num_noise = 1):
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        noise = var_noise * torch.randn([*diff_time.size(), num_noise], device = diff_time.device)
        t_noise = diff_time[:,:,None] + noise
        t_var = t_noise
        t_var = torch.autograd.Variable(t_var, requires_grad=True)

        score = self.get_score(t_var, event_type, event_time, time_gap, non_pad_mask)
        noise_score = -noise / var_noise ** 2

        all_intensity = self.get_intensity(diff_time, event_type, event_time, time_gap, non_pad_mask).squeeze(2)
        sum_intensity  = all_intensity.sum(-1)
        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_time.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_time.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_time.device)

        type_intensity = (all_intensity * type_mask[:, 1:, :]).sum(-1)
        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()

        loss = (0.5 * (score - noise_score) ** 2 * non_pad_mask[:,1:,:]).sum(-1) / num_noise 
        # loss *= var_noise ** 2
        loss += alpha * CELoss * non_pad_mask[:,1:].squeeze(-1)

        return loss
    
    def get_intensity_n_score(self, t, event_type, event_time, time_gap, non_pad_mask):
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        self.affect = self.affect_layer(self.enc_output)
        self.base = self.base_layer(self.enc_output)
        
        intensity = self.intensity_layer(self.affect[:,:-1,None,:] * t.unsqueeze(3) + self.base[:,:-1,None,:]).squeeze(3) # (batch*len-1*1/num_samples)
        all_lambda = intensity * non_pad_mask[:, :-1, None, :]
        intensity_total = all_lambda.sum(-1)

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask[:,1:,:])
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0]*non_pad_mask[:,1:,:]
        score = intensity_total_grad_t - intensity_total

        return all_lambda.squeeze(-2), score.squeeze(-1)
    
            
    def get_intensity(self, t, event_type, event_time, time_gap, non_pad_mask):
        # t size: batch*len-1*num_samples
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        self.affect = self.affect_layer(self.enc_output)
        self.base = self.base_layer(self.enc_output)
        
        intensity = self.intensity_layer(self.affect[:,:-1,None,:] * t.unsqueeze(3) + self.base[:,:-1,None,:]).squeeze(3) # (batch*len-1*1/num_samples)
        all_lambda = intensity * non_pad_mask[:, :-1, None, :]

        
        return all_lambda
    

    def get_score(self, t, event_type, event_time, time_gap, non_pad_mask):
        # t size: batch*len-1*num_samples
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        all_intensity = self.get_intensity(t, event_type, event_time, time_gap, non_pad_mask) # batch*len*num_samples*num_type
        intensity_total = all_intensity.sum(-1)*non_pad_mask[:,1:,:]

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask[:,1:,:])
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0]*non_pad_mask[:,1:,:]
        score = intensity_total_grad_t - intensity_total

        return score


    def compute_loss_mle(self, event_type, event_time, time_gap, non_pad_mask, num_grid = 10):
        non_pad_mask = get_non_pad_mask(event_type)
            
        event_ll, non_event_ll = self.log_likelihood(event_time, time_gap, event_type, num_grid)
        event_loss = -torch.sum(event_ll - non_event_ll)
        loss = event_loss
        return loss
    
    def predict(self, event_type, event_time, time_gap, opt):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        time_gap = torch.cat((event_time[:,0:1], time_gap), axis = 1)
        event_time = torch.concatenate((torch.zeros(event_time.shape[0], 1).to(opt.device), event_time), axis = 1)
        event_type = torch.cat((torch.ones(event_type.shape[0], 1).type(torch.long).to(opt.device), event_type), axis = 1)
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        self.enc_output = enc_output
        intensity_pred = self.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask).squeeze(2)
        _, type_pred = torch.max(intensity_pred, dim=-1)
        return type_pred + 1
    
    def compute_score(self, event_type, event_time, time_gap):
        # this function is only for debug
        non_pad_mask = get_non_pad_mask(event_type)
        self.enc_output = self.encoder(event_type, event_time, non_pad_mask)
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        t_var = torch.autograd.Variable(diff_time, requires_grad=True)

        score = self.get_score(t_var, event_type, event_time, time_gap, non_pad_mask).squeeze(-1)
        score_grad = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0]

        return score, score_grad


    def compute_event(self, event_time, time_gap, event_type, non_pad_mask):
    
        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_time.device)
        for i in range(self.num_types):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(event_time.device)

        all_lambda = self.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask).squeeze(2)

        event = torch.sum(all_lambda * type_mask[:, 1:, :], dim=2)
        event += math.pow(10, -9)
        event.masked_fill_(~non_pad_mask[:,1:].squeeze(2).bool(), 1.0)
        result = torch.log(event+1e-10) * non_pad_mask[:,1:].squeeze(2)
        return result

    def compute_integral_unbiased(self, event_time, time_gap, event_type, non_pad_mask, num_grid):
        """ Log-likelihood of non-events, using Monte Carlo integration. """

        num_samples = num_grid
        if self.normalize == 'log':
            time_low = min(-1.0,time_gap.min()-1.0)
        else:
            time_low = 0
        temp_time = (time_gap.unsqueeze(2) - time_low) * \
                    torch.rand([*time_gap.size(), num_samples], device=event_time.device) + time_low
        
        if self.num_types >= 100:
            all_lambda = None
            for i in range(num_samples):
                lambda_i = self.get_intensity(temp_time[:,:,i:i+1], event_type, event_time, time_gap, non_pad_mask)
                if all_lambda == None:
                    all_lambda = torch.sum(lambda_i, dim=(2,3)) 
                else:
                    all_lambda += torch.sum(lambda_i, dim=(2,3)) 
            all_lambda /= num_samples
        else:
            all_lambda = self.get_intensity(temp_time, event_type, event_time, time_gap, non_pad_mask)
            all_lambda = torch.sum(all_lambda, dim=(2,3)) / num_samples

        unbiased_integral = all_lambda * (time_gap - time_low) * non_pad_mask.squeeze(-1)[:,1:]
        return unbiased_integral


    def log_likelihood(self, event_time, time_gap, event_type, num_grid):
        """ Log-likelihood of sequence. """

        non_pad_mask = get_non_pad_mask(event_type)

        # event log-likelihood
        event_ll = self.compute_event(event_time, time_gap, event_type, non_pad_mask)
        event_ll = torch.sum(event_ll, dim=-1)

        # non-event log-likelihood, either numerical integration or MC integration
        non_event_ll = self.compute_integral_unbiased(event_time, time_gap, event_type, non_pad_mask, num_grid)
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll
        










