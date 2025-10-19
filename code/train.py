import dpp
import numpy as np
import torch
import os
import yaml
import argparse
from copy import deepcopy
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def aggregate_loss_over_dataloader(model, dl):
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dl:
            total_loss += -model.log_prob(batch).sum()
            total_count += batch.size
    return total_loss / total_count

def train_model(config):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset_name = config['dataset_name']
    model_save_path = config['model_save_path']
    model_type = config['model_type']
    

    batch_size = config['training']['batch_size']
    regularization = config['training']['regularization']
    learning_rate = config['training']['learning_rate']
    max_epochs = config['training']['max_epochs']
    patience = config['training']['patience']

    dataset = dpp.data.load_dataset(dataset_name)
    d_train, d_val, d_test = dataset.train_val_test_split(seed=seed)

    dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
    dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)

    mean_log_inter_time, std_log_inter_time = d_test.get_inter_time_statistics()
    
    model = dpp.models.LogNormMixTransformer(
        num_marks=d_train.num_marks,
        encoder_type=model_type,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        context_size=config['model']['context_size'],
        num_mix_components=config['model']['num_mix_components'],

        nhead=config['model']['transformer']['nhead'],
        num_layers=config['model']['transformer']['num_layers'],
        dropout=config['model']['transformer']['dropout']
    )

    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)
    num_params = sum(p.numel() for p in model.parameters())

    if '/' in dataset_name:
        base_dataset_name = dataset_name.split('/')[-1]
    else:
        base_dataset_name = dataset_name

    print(f'Building {model_type} model on {base_dataset_name} dataset, num_marks={d_train.num_marks}, num_params={num_params}')
    print(f'Batches in train: {len(dl_train)}, val: {len(dl_val)}, test: {len(dl_test)}')
    print('Starting training...')

    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses = []

    for epoch in range(max_epochs):
        model.train()
        for batch in dl_train:
            opt.zero_grad()
            loss = -model.log_prob(batch).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            loss_val = aggregate_loss_over_dataloader(model, dl_val)
            training_val_losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            if impatient > 0:
                print(f"Epoch {epoch:4d}: train_loss = {loss.item():.4f}, val_loss = {loss_val:.4f}, impatience set to 0, saving model")
            else:
                print(f"Epoch {epoch:4d}: train_loss = {loss.item():.4f}, val_loss = {loss_val:.4f}, saving model")
            impatient = 0
            best_model = deepcopy(model.state_dict())
            model.save_model(model_save_path, num_params, base_dataset_name)
        else:
            impatient += 1
            print(f"Epoch {epoch:4d}: train_loss = {loss.item():.4f}, val_loss = {loss_val:.4f}, impatience = {impatient}/{patience}")

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

    model.load_state_dict(best_model)
    model.eval()

    with torch.no_grad():
        final_loss_train = aggregate_loss_over_dataloader(model, dl_train)
        final_loss_val = aggregate_loss_over_dataloader(model, dl_val)
        final_loss_test = aggregate_loss_over_dataloader(model, dl_test)

    print(f'Negative log-likelihood:\n'
        f' - Train: {final_loss_train:.4f}\n'
        f' - Val:   {final_loss_val:.4f}\n'
        f' - Test:  {final_loss_test:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a point process model')
    parser.add_argument('--config', type=str, default='scripts/train_config_inhomo_poi.yaml', help='Configuration file path')
    # parser.add_argument('--config', type=str, default='scripts/train_config_multi_hawkes.yaml', help='Configuration file path')
    # parser.add_argument('--config', type=str, default='scripts/train_config_self_correct.yaml', help='Configuration file path')
    # parser.add_argument('--config', type=str, default='scripts/train_config_hawkes.yaml', help='Configuration file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    train_model(config)