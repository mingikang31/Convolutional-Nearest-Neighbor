'''Dataset File for MNIST1D and other for CNN'''

import torch
import numpy as np
from mnist1d.data import make_dataset


class MNIST1D_Dataset():
   
   def __init__(self, seed = None): 

      
      self.data_args = self.get_dataset_args(as_dict=False)
      
      self.model_args = self.get_model_args(as_dict=False)
      
      
      if not seed: 
         self.set_seed(self.data_args.seed)
      else: 
         self.set_seed(seed)
         
      print("The Arguments for Data are: ")
      print("num_samples: 5000 \n train_split: 0.8 \n template_len: 12 \n padding: [36,60] \n scale_coeff: .4 \n max_translation: 48 \n corr_noise_scale: 0.25 \n iid_noise_scale: 2e-2 \n shear_scale: 0.75 \n shuffle_seq: False \n final_seq_length: 40 \n seed: 42")
      
      print("\n")
      
      print("The Arguments for Model are: ")
      print("input_size: 40 \n output_size: 10 \n hidden_size: 256 \n learning_rate: 1e-2 \n weight_decay: 0 \n batch_size: 100 \n total_steps: 6000 \n print_every: 1000 \n eval_every: 250 \n checkpoint_every: 1000 \n device: mps \n seed: 42")


   
   def make_dataset(self): 
      data = make_dataset(self.data_args)
      # Creating dataset of size [Batch, channels, tokens]
      data['x'] = torch.Tensor(data['x']).unsqueeze(1)
      data['x_test'] = torch.Tensor(data['x_test']).unsqueeze(1)
      return data
   
   @staticmethod
   def set_seed(seed):
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      
   def get_dataset_args(self, as_dict=False):
      arg_dict = {'num_samples': 5000,
               'train_split': 0.8,
               'template_len': 12,
               'padding': [36,60],
               'scale_coeff': .4, 
               'max_translation': 48,
               'corr_noise_scale': 0.25,
               'iid_noise_scale': 2e-2,
               'shear_scale': 0.75,
               'shuffle_seq': False,
               'final_seq_length': 40,
               'seed': 42}
      return arg_dict if as_dict else self.ObjectView(arg_dict)
   
   def get_model_args(self, as_dict=False):
      arg_dict = {'input_size': 40,
               'output_size': 10,
               'hidden_size': 256,
               'learning_rate': 1e-2,
               'weight_decay': 0,
               'batch_size': 100,
               'total_steps': 6000,
               'print_every': 1000,
               'eval_every': 250,
               'checkpoint_every': 1000,
               'device': 'mps',
               'seed': 42}
      return arg_dict if as_dict else self.ObjectView(arg_dict)
   
   @staticmethod
   class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
      
      
      
'''Example Usage '''
Data_model = MNIST1D_Dataset()
# Data_model.data_args.corr_noise_scale = 0.00
# Data_model.data_args.iid_noise_scale = 0.00
# data = Data_model.make_dataset()

# print(Data_model.data_args)
# print(Data_model.model_args)

# print(data['x'].shape)