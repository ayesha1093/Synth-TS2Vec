# partially adopted from https://github.com/proceduralia/pytorch-GAN-timeseries/tree/8e7d62fed6f4061d13ec9dfd84e07520d4257ed2

import torch
import numpy as np
import argparse
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import torch.nn.init as init
import time
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save
from pathlib import Path

# Set up the argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output, and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The GPU index to use for training and inference (defaults to 0)')
    parser.add_argument('--epochs', type=int, default=50, help='The number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate for training')
    parser.add_argument('--seed', type=int, default=None, help='The random seed for initialization')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum number of threads to use')

    args = parser.parse_args()
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    run_dir = 'GANtraining/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    device = init_dl_program('cpu', seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)

    dataset_name = args.dataset
    run_name = args.run_name
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    gpu_index = args.gpu

    class LSTMGenerator(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256, leaky_relu_slope=0.1):
            super(LSTMGenerator, self).__init__()
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
            self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
            self.linear = nn.Linear(hidden_dim, out_dim)
            self.init_weights()

        def init_weights(self):
            for name, param in self.named_parameters():
                if 'weight' in name:
                    if 'lstm' in name:
                        init.xavier_normal_(param)
                    elif 'linear' in name:
                        init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.constant_(param, 0.0)

        def forward(self, input):
            batch_size, seq_len = input.size(0), input.size(1)
            h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(input.device)
            c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(input.device)
            recurrent_features, _ = self.lstm(input, (h_0, c_0))
            recurrent_features = self.leaky_relu(recurrent_features)
            outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
            outputs = outputs.view(batch_size, seq_len, self.out_dim)
            return outputs


    class LSTMDiscriminator(nn.Module):
        def __init__(self, in_dim, n_layers=1, hidden_dim=256, leaky_relu_slope=0.1):
            super(LSTMDiscriminator, self).__init__()
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
            self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
            self.linear = nn.Linear(hidden_dim, 1)

        def forward(self, input):
            batch_size, seq_len = input.size(0), input.size(1)
            h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(input.device)
            c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(input.device)
            recurrent_features, _ = self.lstm(input, (h_0, c_0))
            recurrent_features = self.leaky_relu(recurrent_features)
            outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
            outputs = outputs.view(batch_size, seq_len, 1)
            outputs = outputs.mean(dim=1).view(-1, 1) 
            return outputs

    def compute_gradient_penalty(netD, real_samples, fake_samples):
        epsilon = torch.rand(real_samples.size(0), 1, 1).to(real_samples.device)
        interpolated = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
        critic_scores = netD(interpolated)
        gradients = torch.autograd.grad(outputs=critic_scores, inputs=interpolated,
                                        grad_outputs=torch.ones_like(critic_scores),
                                        create_graph=True, retain_graph=True)[0]
        lambda_gp = 10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return gradient_penalty

    n_critic = 5
    seq_len = train_data.shape[1]
    num_features = train_data.shape[-1]
    noise_dim = 100
    n_layers = 1
    hidden_dim = 256

    netG = LSTMGenerator(in_dim=noise_dim, out_dim=num_features, n_layers=n_layers, hidden_dim=hidden_dim)
    netD = LSTMDiscriminator(in_dim=num_features, n_layers=n_layers, hidden_dim=hidden_dim)
    print(netG)
    print(netD)
    
    netG.to(device)
    netD.to(device)

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)
    best_loss = float('inf')
    no_improvement = 0
    patience = 10

    for epoch in range(num_epochs):
        epoch_critic_loss = 0.0
        epoch_generator_loss = 0.0

        for i, (real_samples, _) in enumerate(train_loader):
            # Train Discriminator
            for _ in range(n_critic):
                netD.zero_grad()
                real_samples = real_samples.to(device)
                noise = torch.randn(real_samples.size(0), seq_len, noise_dim, device=device)
                fake_samples = netG(noise).detach()
                critic_loss = -netD(real_samples).mean() + netD(fake_samples).mean()
                gradient_penalty = compute_gradient_penalty(netD, real_samples, fake_samples)
                critic_loss += gradient_penalty
                critic_loss.backward()
                optimizerD.step()
                epoch_critic_loss += critic_loss.item()

            # Train Generator
            netG.zero_grad()
            noise = torch.randn(real_samples.size(0), seq_len, noise_dim, device=device)
            fake_samples = netG(noise)
            generator_loss = -torch.mean(netD(fake_samples))
            generator_loss.backward()
            optimizerG.step()
            epoch_generator_loss += generator_loss.item()

        epoch_critic_loss /= len(train_loader)
        epoch_generator_loss /= len(train_loader)
        schedulerG.step()
        schedulerD.step()
        print(f'Epoch {epoch+1}/{num_epochs} - Critic Loss: {epoch_critic_loss}, Generator Loss: {epoch_generator_loss}')

        
        torch.save(netG.state_dict(), os.path.join(run_dir, 'generator_state.pth'))
        torch.save(netD.state_dict(), os.path.join(run_dir, 'discriminator_state.pth'))
        

    num_samples = train_data.shape[0]
    fixed_noise = torch.randn(num_samples, seq_len, noise_dim, device=device)
    with torch.no_grad():
        generated_sequences = netG(fixed_noise).detach().cpu()

    generated_samples_dir = os.path.join(run_dir, 'generated_samples')
    os.makedirs(generated_samples_dir, exist_ok=True)
    generated_samples_file = os.path.join(generated_samples_dir, 'generated_samples.npy')
    np.save(generated_samples_file, generated_sequences.numpy())

    generated_sequences_2d = generated_sequences.reshape((generated_sequences.shape[0], -1))
    train_data_2d = train_data.reshape((train_data.shape[0], -1))

def visualization(ori_data, generated_data, analysis='pca', dataset_name=None, batch_size=None, learning_rate=None, save_path=None):
    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx = np.random.permutation(min(len(ori_data), len(generated_data)))[:anal_sample_no]
    ori_subsample = np.asarray(ori_data)[idx]
    gen_subsample = np.asarray(generated_data)[idx]
    combined_data = np.concatenate([ori_subsample, gen_subsample], axis=0)

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(combined_data)
        plt.figure(figsize=(8, 8))
        plt.scatter(pca_results[:anal_sample_no, 0], pca_results[:anal_sample_no, 1],
                    c='red', label='Original', alpha=0.5)
        plt.scatter(pca_results[anal_sample_no:, 0], pca_results[anal_sample_no:, 1],
                    c='blue', label='Synthetic', alpha=0.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA Results: {dataset_name}, LR: {learning_rate}, Batch: {batch_size}')
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, 'pca_results.png'))
        plt.show()

    elif analysis == 'tsne':
        perplexity_value = min(30, anal_sample_no - 1)
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
        tsne_results = tsne.fit_transform(combined_data)
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c='red', label='Original', alpha=0.5)
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c='blue', label='Synthetic', alpha=0.5)
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.title(f't-SNE Results: {dataset_name}, LR: {learning_rate}, Batch: {batch_size}')
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, 'tsne_results.png'))
        plt.show()

    elif analysis == 'histogram':
        plt.figure(figsize=(12, 6))
        plt.hist(ori_subsample.flatten(), bins=50, alpha=0.5, label='Original', color='red')
        plt.hist(gen_subsample.flatten(), bins=50, alpha=0.5, label='Synthetic', color='blue')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram: {dataset_name}, LR: {learning_rate}, Batch: {batch_size}')
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, 'histogram.png'))
        plt.show()

    elif analysis == 'kde':
        plt.figure(figsize=(12, 6))
        sns.kdeplot(ori_subsample.flatten(), color='red', label='Original', fill=True, alpha=0.5)
        sns.kdeplot(gen_subsample.flatten(), color='blue', label='Synthetic', fill=True, alpha=0.5)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'KDE: {dataset_name}, LR: {learning_rate}, Batch: {batch_size}')
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, 'kde.png'))
        plt.show()

    else:
        raise ValueError("Analysis must be 'pca', 'tsne', 'histogram', or 'kde'")
