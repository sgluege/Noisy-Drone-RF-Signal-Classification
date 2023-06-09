import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

dataset_folder = 'dataset/'

class DroneSignalsDatasetIQandSpec(Dataset):
    """
    Class for custom dataset of drone data comprised of
    x_iq (torch.tensor.float): signals iq data(n_samples x 2 x input_vec_length)
    x_spec (torch.tensor.float): signals spectogram (n_samples x 2 x num_segments x num_segments)
    y (torch.tensor.long): targets (n_samples)
    snrs (torch.tensor.int): SNRs per sample (n_samples) 
    duty_cycle (torch.tensor.float): duty cycle length per sample (n_samples) 
    Args:
        Dataset (torch tensor): 
    """
    def __init__(self, x_iq_tensor, x_spec_tensor, y_tensor, snr_tensor, duty_cycle_tensor):
        self.x_iq = x_iq_tensor
        self.x_spec = x_spec_tensor
        self.y = y_tensor
        self.snr = snr_tensor
        self.dury_cyle = duty_cycle_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_iq[idx], self.x_spec[idx], self.y[idx], self.snr[idx], self.dury_cyle[idx]   

    def targets(self):
        return self.y 

    def snrs(self):
        return self.snr

    def duty_cycle(self):
        return self.duty_cycle


def plot_input_data(spectrogram_2d, iq_2d, title='', figsize=(10,9)):

    fig, axs = plt.subplot_mosaic([['spec_re', 'spec_im'], ['spec_re', 'spec_im'], ['iq_re', 'iq_re'], ['iq_im', 'iq_im']], figsize=figsize) # layout='constrained'
    
    # plot spectrogram Re and Im
    spec_re = axs['spec_re'].imshow(spectrogram_2d[0,:,:]) #, aspect='auto', origin='lower')
    axs['spec_re'].set_title('Re', fontsize=10)
    fig.colorbar(spec_re, ax=axs['spec_re'], location='right', shrink=0.5)

    spec_im = axs['spec_im'].imshow(spectrogram_2d[1,:,:]) #, aspect='auto', origin='lower')
    axs['spec_im'].set_title('Im', fontsize=10)
    fig.colorbar(spec_im, ax=axs['spec_im'], location='right', shrink=0.5)

    # plot iq Re and Im
    axs['iq_re'].plot(iq_2d[0,:])
    axs['iq_re'].set_title('IQ data')
    axs['iq_re'].set_ylabel('Re', rotation=0)

    axs['iq_im'].plot(iq_2d[1,:])
    # axs['iq_im'].set_title('Im')
    axs['iq_im'].set_xlabel('Time (samples)')
    axs['iq_im'].set_ylabel('Im', rotation=0)
    
    # add figure title
    fig.suptitle(plt_title + '\n\nSpectrogram')
    plt.savefig('sample_input_data.png', dpi=300, bbox_inches='tight')   
    # plt.show()


# read statistics/class count of the dataset
dataset_stats = pd.read_csv(dataset_folder + 'class_stats.csv', index_col=0)
class_names = dataset_stats['class'].values

# read SNR count of the dataset
snr_stats = pd.read_csv(dataset_folder + 'SNR_stats.csv', index_col=0)
snr_list = snr_stats['SNR'].values

# load data
dataset_dict = torch.load(dataset_folder + 'dataset.pt')
dataset_dict.keys()

x_iq = dataset_dict['x_iq']
x_spec = dataset_dict['x_spec']
y = dataset_dict['y']
snrs = dataset_dict['snr']
duty_cycle = dataset_dict['duty_cycle']

# create pytorch dataset form tensors
dataset = DroneSignalsDatasetIQandSpec(x_iq,x_spec,y,snrs,duty_cycle)
del(x_iq, x_spec, y, snrs, duty_cycle, dataset_dict)

# define a data loader
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=64)

# get a batch of samples from the data loader
x_iq, x_spec, labels, snrs, duty_cycle =  next(iter(data_loader))
# x_iq.shape, x_spec.shape, labels.shape, snrs.shape, duty_cycle.shape
print('Loaded a batch of samples from the data loader with batch size', x_iq.shape[0], 'and the following shapes:')
print('x_iq shape: ', x_iq.shape)
print('x_spec shape: ', x_spec.shape)
print('labels shape: ', labels.shape)
print('snrs shape: ', snrs.shape)
print('duty_cycle shape: ', duty_cycle.shape)


# select a sample from the batch
sample_id = 12

# plot the sample
act_snr = snrs[sample_id]
act_duty_cycle = duty_cycle[sample_id]
act_class = class_names[labels[sample_id]]
plt_title = 'Class: ' + act_class + ', SNR: ' + str(act_snr.numpy()) + 'dB, Duty Cycle: ' + str(act_duty_cycle.numpy())

spectrogram_2d = x_spec[sample_id,:,:,:]
iq_2d = x_iq[sample_id,:,:]
plot_input_data(spectrogram_2d, iq_2d, title=plt_title, figsize=(10,7))
