import sys, os
import numpy as np
import pandas as pd
import time
import tqdm
from scipy.special import erfinv
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler # for sampling the data for training and validation separately
from . import VAE_encoder, VAE_decoder

class VAE_model(nn.Module):
    """
    Class for the VAE model with estimation of weights distribution parameters via Mean-Field VI.
    """
    def __init__(self,
            model_name,
            data,
            encoder_parameters,
            decoder_parameters,
            training_parameters,
            random_seed,
            pretrained_model = False
            ):
        
        super().__init__()
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 # data type of entries in the input data tensor
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        
        self.num_species = data.num_species
        self.alphabet_size = data.alphabet_size

        self.encoder_parameters=encoder_parameters
        self.decoder_parameters=decoder_parameters

        encoder_parameters['alphabet_size'] = self.alphabet_size
        decoder_parameters['alphabet_size'] = self.alphabet_size
        self.training_parameters = training_parameters
        
        self.encoder = VAE_encoder.VAE_MLP_encoder(params=encoder_parameters)
        if decoder_parameters['bayesian_decoder']:
            self.decoder = VAE_decoder.VAE_Bayesian_MLP_decoder(params=decoder_parameters)
        else:
            self.decoder = VAE_decoder.VAE_Standard_MLP_decoder(params=decoder_parameters)
        self.logit_sparsity_p = decoder_parameters['logit_sparsity_p']
        self.pretrained_model = pretrained_model
        self.start_epoch = 1
        self.best_val_loss = float('inf')

    def get_train_valid_loader(self, data):
        """
        Function copied from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py-L14
        and then modified (simplified) by Ha Vu on 01/17/2023
        Utility function for loading and returning train and valid
        multi-process iterators over the CIFAR-10 dataset. A sample
        9x9 grid of the images can be optionally displayed.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data: the data object from data_utils.py
        Returns
        -------
        - train_loader: training set iterator.
        - valid_loader: validation set iterator.
        """
        # load the dataset
        num_train = len(data)
        indices = list(range(num_train))
        split = int(np.floor(self.training_parameters['validation_set_fract'] * num_train))

        np.random.seed(self.random_seed)
        np.random.shuffle(indices) # this will change indices, inplace

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(data, batch_size=self.training_parameters['batch_size'], sampler=train_sampler, num_workers=0)
        valid_loader = DataLoader(data, batch_size=self.training_parameters['batch_size'], sampler=valid_sampler, num_workers=0)
        # to fix this error: https://stackoverflow.com/questions/60101168/pytorch-runtimeerror-dataloader-worker-pids-15332-exited-unexpectedly
        return (train_loader, valid_loader)

    def load_pretrained_model(self, pretrained_model_fn):
        self.pretrained_model = True
        try:
            checkpoint = torch.load(pretrained_model_fn, map_location =  self.device)
            self.load_state_dict(checkpoint['model_state_dict']) # function load_state_dict is part of nn.Module
            print("Initialized VAE with checkpoint '{}' ".format(pretrained_model_fn))
        except:
            print("Unable to locate VAE model checkpoint")
            sys.exit(0)

    def set_training_params(self):
        '''
        Some attributes of the object that will be declared in this function:
        self.log_fn
        self.optimizer
        self.scheduler
        '''
        if self.training_parameters['log_training_info']:
            self.log_fn = self.training_parameters['training_logs_location']+os.sep+self.model_name+"_losses.csv"
            with open(self.log_fn, "a") as logs:
                logs.write("Number of bp in each batch:\t"+str(self.training_parameters['batch_size'])+"\n")
        else:
            self.log_fn = None

        self.optimizer = optim.Adam(self.parameters(), lr=self.training_parameters['learning_rate'], weight_decay = self.training_parameters['l2_regularization'])

        if self.training_parameters['use_lr_scheduler']:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=training_parameters['lr_scheduler_step_size'], gamma=training_parameters['lr_scheduler_gamma'])
        else:
            self.scheduler = None


    def sample_latent(self, mu, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu).to(self.device)
        z = torch.exp(0.5*log_var) * eps + mu
        return z

    def KLD_diag_gaussians(self, mu, logvar, p_mu, p_logvar):
        """
        KL divergence between diagonal gaussian with prior diagonal gaussian.
        """
        KLD = 0.5 * (p_logvar - logvar) + 0.5 * (torch.exp(logvar) + torch.pow(mu-p_mu,2)) / (torch.exp(p_logvar)+1e-20) - 0.5
        return torch.sum(KLD)

    def annealing_factor(self, annealing_warm_up, epoch):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        if epoch < annealing_warm_up:
            return epoch/annealing_warm_up
        else:
            return 1

    def KLD_global_parameters(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        KLD_decoder_params = 0.0
        zero_tensor = torch.tensor(0.0).to(self.device) 
        
        for layer_index in range(len(self.decoder.hidden_layers_sizes)):
            for param_type in ['weight','bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['hidden_layers_mean.'+str(layer_index)+'.'+param_type].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['hidden_layers_log_var.'+str(layer_index)+'.'+param_type].flatten(),
                                    zero_tensor,
                                    zero_tensor
                )
                
        for param_type in ['weight','bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                        self.decoder.state_dict(keep_vars=True)['last_hidden_layer_'+param_type+'_mean'].flatten(),
                                        self.decoder.state_dict(keep_vars=True)['last_hidden_layer_'+param_type+'_log_var'].flatten(),
                                        zero_tensor,
                                        zero_tensor
                )

        if self.decoder.include_sparsity:
            self.logit_scale_sigma = 4.0
            self.logit_scale_mu = 2.0**0.5 * self.logit_scale_sigma * erfinv(2.0 * self.logit_sparsity_p - 1.0)

            sparsity_mu = torch.tensor(self.logit_scale_mu).to(self.device) 
            sparsity_log_var = torch.log(torch.tensor(self.logit_scale_sigma**2)).to(self.device)
            KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['sparsity_weight_mean'].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['sparsity_weight_log_var'].flatten(),
                                    sparsity_mu,
                                    sparsity_log_var
            )
            
        if self.decoder.convolve_output:
            for param_type in ['weight']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['output_convolution_mean.'+param_type].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['output_convolution_log_var.'+param_type].flatten(),
                                    zero_tensor,
                                    zero_tensor
                )

        if self.decoder.include_temperature_scaler:
            KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['temperature_scaler_mean'].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['temperature_scaler_log_var'].flatten(),
                                    zero_tensor,
                                    zero_tensor
            )        
        return KLD_decoder_params

    def loss_function(self, x_recon_log, x, mu, log_var, kl_latent_scale, kl_global_params_scale, annealing_warm_up, epoch):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        Note: 
        mu and log_var both have dimensions (batch_size, z_dim) --> variance is diagonal
        log_var = log(sigma^2 * I)
        """
        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction='sum') / x.shape[0] # first use sigmoid to convert x_recon_log to [0,1], and apply binary cross entropy to compare with X, ref: https://zhang-yang.medium.com/how-is-pytorchs-binary-cross-entropy-with-logits-function-related-to-sigmoid-and-d3bd8fb080e7
        # goal of an optimizer is to minimize the  binary_cross_entropy_with_logits, because the binary cross entropy here functions as negative log likelihood for each data point of the matrix (#batch_size,#species,#bp)
        # BCE is the average binary cross entroy for each genomic position (summing over the -log(P) of each entry in the matrix (#species,#bp))
        # a number (already averaged across batch/animal)
        # this term is actually equal to P (X | Z), where x_recon_log represents the parameters of distribution of X|Z
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0] # KL divergence between Normal(mu, log_var) and normal(0,I). Note log_var: (batch_size, z_dim) and has values of log of sigma^2
        # Note: torch.sum with no specification of the dimension will sum over all entries  of the matrix 
        if self.decoder.bayesian_decoder:
            KLD_decoder_params_normalized = self.KLD_global_parameters() / x.shape[0]
        else:
            KLD_decoder_params_normalized = 0.0
        warm_up_scale = self.annealing_factor(annealing_warm_up,epoch)
        neg_ELBO = BCE + warm_up_scale * (kl_latent_scale * KLD_latent + kl_global_params_scale * KLD_decoder_params_normalized)
        # warm_up_scale is there so that the beginning of training focuses on succesfully reconstructing the observed data
        return neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized
    
    def all_likelihood_components(self, x):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        x: (# mutated_seq, num_species, alphabet_len)
        """
        mu, log_var = self.encoder(x) # (#mutated_seq, k), K: hidden dimension
        z = self.sample_latent(mu, log_var) # (#mutated_seq, k), k: hidden dimension
        recon_x_log = self.decoder(z) # (# mutated_seq, num_species, alphabet_len)

        recon_x_log = recon_x_log.view(-1,self.alphabet_size*self.num_species) # (#mutated_seq, num_species*alphabet_len)
        x = x.view(-1,self.alphabet_size*self.num_species) # (#mutated_seq, num_species*alphabet_len)
        
        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'),dim=1) # sum over the second dimension --> (#mutated_seq); each value is the sum of BCE over num_species*alphabet_len
        # let's break this down: 
        # if t = F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none') --> t: (#mutated_seq, num_species*alphabet_len) --> same shape as recon_x_log and x
        # (Pdb) recon_x_log[0,:3]
        # tensor([-5.8681, -6.2809, -6.6955], device='cuda:0')
        # (Pdb) x[0,:3]
        # tensor([0., 0., 0.], device='cuda:0')
        # (Pdb) t[0,:3]
        # tensor([0.0028, 0.0019, 0.0012], device='cuda:0')
        # (Pdb) def sigmoid(x): return (1 + (-x).exp()).reciprocal()
        # (Pdb) logit_pred = recon_x_log[0,:3]
        # (Pdb) truth = x[0,:3]
        # (Pdb) pred_prob = sigmoid(logit_pred)
        # -truth * pred_prob.log() - (1-truth) * (1-pred_prob).log() # this is the binary cross entropy, and it is equal to t
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1))
        
        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def report_progress(self, epoch, neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized, start):
        '''
        During training, this function is called to report on the progress of training and save the model if necessary
        '''
        if epoch % self.training_parameters['log_training_freq'] == 0:
            progress = "|Train : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(epoch, neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized, time.time() - start)
            print(progress)

            if self.training_parameters['log_training_info']:
                with open(self.log_fn, "a") as logs:
                    logs.write(progress+"\n")

        if epoch % self.training_parameters['save_model_params_freq']==0:
            self.save(model_checkpoint=self.training_parameters['model_checkpoint_folder']+os.sep+self.model_name+"_step_"+str(epoch),
                        encoder_parameters=self.encoder_parameters,
                        decoder_parameters=self.decoder_parameters,
                        training_parameters=self.training_parameters)

    def validate_model(self, valid_loader, epoch, start):
        self.eval()
        with torch.no_grad():
            for batch_index, x in enumerate(valid_loader, 0):
                x = torch.tensor(x, dtype=self.dtype).to(self.device) #  batch, bp, alphabet
                mu, log_var = self.encoder(x)
                z = self.sample_latent(mu, log_var)
                recon_x_log = self.decoder(z)
                neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = self.loss_function(recon_x_log, x, mu, log_var, kl_latent_scale = self.training_parameters['kl_latent_scale'], kl_global_params_scale = self.training_parameters['kl_global_params_scale'], annealing_warm_up = self.training_parameters['annealing_warm_up'], epoch = 1) # the reported epoch is 1, which means that, due to how the annealing factos are set up, the validation will try to prioritize models that show the best reconstruction of input data
                progress_val = "\t\t\t|Val : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, Time: {4:.2f} |".format(epoch, neg_ELBO.item(), BCE.item(), KLD_latent.item(), time.time() - start)
                print(progress_val)
                if neg_ELBO.item() < self.best_val_loss:
                    self.best_val_loss =  neg_ELBO.item()
                    self.save(model_checkpoint=self.training_parameters['e']+os.sep+self.model_name+"_best",
                                encoder_parameters=self.encoder_parameters,
                                decoder_parameters=self.decoder_parameters,
                                training_parameters=self.training_parameters)
                return # this meanns that we will onnly look at one batch for validation


    def train_model(self, data):
        """
        Training procedure for the VAE model.
        If use_validation_set is True then:
            - we split the alignment data in train/val sets.
            - we train up to num_epochs steps but store the version of the model with lowest loss on validation set across training
        If not, then we train the model for num_epochs and save the model at the end of training
        """
        if torch.cuda.is_available():
            cudnn.benchmark = True
        self.train() # the train() function may be a function of nn.Module
        
        self.set_training_params()
        
        train_loader, valid_loader = self.get_train_valid_loader(data)
        start = time.time()
        train_loss = 0
        for epoch in tqdm.tqdm(range(self.start_epoch ,self.training_parameters['num_epoch']+1), desc="Training model"):
            for batch_index, x in enumerate(train_loader, 0): # each time we enumerate data loader, dataloader actually reshuffle at each epoch
                x = torch.tensor(x, dtype=self.dtype).to(self.device) #  batch, bp, alphabet
                self.optimizer.zero_grad() # set gradients back to 0 for every trainging iteration. Ref: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                mu, log_var = self.encoder(x) # mu, log_var has the same shape: [batch_size (animal), z-dim]
                z = self.sample_latent(mu, log_var) # z shape: [batch_size (animal), z-dim]
                recon_x_log = self.decoder(z) # recon_x_log shape: [batch_size (animal), #letter_in_seq, #letter_in_alphabet]
                
                neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = self.loss_function(recon_x_log, x, mu, log_var, self.training_parameters['kl_latent_scale'], self.training_parameters['kl_global_params_scale'], self.training_parameters['annealing_warm_up'], epoch)
                
                neg_ELBO.backward() # this is required for everything iteration of training
                self.optimizer.step() # also required for every iteration of training
                
                if self.training_parameters['use_lr_scheduler']:
                    self.scheduler.step()
                
                if self.training_parameters['use_validation_set'] and epoch % self.training_parameters['validation_freq'] == 0:
                    neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = self.validate_model(valid_loader, epoch, start)

                self.report_progress(epoch, neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized, start)
                if batch_index == (self.training_parameters['num_batch_per_epoch'] - 1):
                    break
        return

    def continue_train_model(self, data, pretrained_model_fn, start_epoch):
        '''
        If we have trained a model and it stopped because it ran out of training time (computing cluster limits)
        We would like to load the model up until the existing path and continue training where it left off
        '''
        self.load_pretrained_model(pretrained_model_fn)
        self.start_epoch = start_epoch
        self.train_model(data)
        return

    

    def save(self, model_checkpoint, encoder_parameters, decoder_parameters, training_parameters, batch_size=256):
        torch.save({
            'model_state_dict':self.state_dict(),
            'encoder_parameters':encoder_parameters,
            'decoder_parameters':decoder_parameters,
            'training_parameters':training_parameters,
            }, model_checkpoint)
    
    def compute_ELBO(self, msa_data, num_samples, batch_size=256):
        """
        msa_data is the object declared in data_utils.py   
        """
        dataloader = torch.utils.data.DataLoader(msa_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        prediction_matrix = torch.zeros(msa_data.msa_df.shape[0],num_samples) # (possible mutations, num_samples). Note possible mutations imply num_genomic_positions * 4 (ACTG)
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
                x = batch.type(self.dtype).to(self.device)
                for j in tqdm.tqdm(range(num_samples), 'Looping through number of samples for batch #: '+str(i+1)):
                    seq_predictions, _, _ = self.all_likelihood_components(x)
                    prediction_matrix[i*batch_size:i*batch_size+len(x),j] = seq_predictions
                tqdm.tqdm.write('\n')
            mean_predictions = prediction_matrix.mean(dim=1, keepdim=False) # (possible mutations)
            std_predictions = prediction_matrix.std(dim=1, keepdim=False) # (possible mutations)
        return  mean_predictions, std_predictions

    def compute_ELBO_for_variants_data(self, msa_data, num_samples, batch_size = 256):
        """
        msa_data is the object declared in data_utils.py   
        This function works based on the assumption that msa_data has an attribute msa_df that already created data of MSA for all the possible genetic variants     
        """
        mean_elbos, _ = self.compute_ELBO(msa_data, num_samples, batch_size) # reason why we make this a separate function from compute_ELBO is because so that compute_ELBO does not have to assume anything about msa_data.msa_df. This function assumes that every 4 rows of msa_data.msa_df is the same except for the column corresponding to the reference genome (hg19). Therefore, this function will add a few lines of code to transform the data into the right form (num_genome_position, 4 (ACTG))
        num_var_per_pos = len(msa_data.refGen_alphabet) # 4
        mean_elbos = mean_elbos.reshape((num_var_per_pos, msa_data.num_genomic_position)).transpose(0,1) # (num_genomic_position, 4 (ACTG))
        return mean_elbos.detach().cpu().numpy()

    def compute_reconstruction_prob_for_variants_data(self, msa_data, num_samples, batch_size = 256):
        '''
        msa_data is the object declared in data_utils.py           
        '''
        dataloader = torch.utils.data.DataLoader(msa_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        num_var_per_pos = len(msa_data.refGen_alphabet) # 4
        prediction_matrix = torch.zeros(msa_data.msa_df.shape[0], num_var_per_pos, num_samples) # (possible mutations, num_samples). Note possible mutations imply num_genomic_positions * 4 (ACTG)
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
                x = batch.type(self.dtype).to(self.device) # (# seq, species, alphabet_len)
                for j in tqdm.tqdm(range(num_samples), 'Looping through number of samples for batch #: '+str(i+1)):
                    mu, log_var = self.encoder(x) # (# seq, k)
                    z = self.sample_latent(mu, log_var) # (#seq, species, alphabet_len)
                    recon_x_log = self.decoder(z) # (# mutated_seq, num_species, alphabet_len)
                    prediction_matrix[i*batch_size:i*batch_size+len(x),:,j] = torch.sigmoid(recon_x_log[:,msa_data.reference_species_index,:num_var_per_pos])
            mean_predictions = prediction_matrix.mean(dim=2, keepdim=False)
        return mean_predictions.detach().cpu().numpy()