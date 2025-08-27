import torch

import matplotlib.pyplot as plt 
import numpy as np 
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import pickle
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class MNIST1D_Dataset(Dataset):
    def __init__(self, train=True):
        
        self.transform = transforms.ToTensor()

        self.args = get_dataset_args() 
        
        self.train = train
        self.data = make_dataset(args=self.args)
        self.x = self.data['x'] if train else self.data['x_test']
        self.y = self.data['y'] if train else self.data['y_test']
        self.x = torch.Tensor(self.x).unsqueeze(1)
        self.y = torch.LongTensor(self.y)
        self.t = self.data['t'] 
        self.templates = self.data['templates']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



""" https://github.com/greydanus/mnist1d/tree/master"""
# The MNIST-1D dataset | 2020
# Sam Greydanus
def get_dataset_args(as_dict=False):
    """ Generate dictionary with dataset properties

    Parameters
    ----------
    as_dict : bool, optional
        if true, return the dataset properties as dictionary; if false, return an ObjectView, by default False

    Returns
    -------
    _type_
        _description_
    """
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
            'seed': 42,
            'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'}
    return arg_dict if as_dict else ObjectView(arg_dict)


# basic 1D templates for the 10 digits
def get_templates():
    d0 = np.asarray([5,6,6.5,6.75,7,7,7,7,6.75,6.5,6,5])
    d1 = np.asarray([5,3,3,3.4,3.8,4.2,4.6,5,5.4,5.8,5,5])
    d2 = np.asarray([5,6,6.5,6.5,6,5.25,4.75,4,3.5,3.5,4,5])
    d3 = np.asarray([5,6,6.5,6.5,6,5,5,6,6.5,6.5,6,5])
    d4 = np.asarray([5,4.4,3.8,3.2,2.6,2.6,5,5,5,5,5,5])
    d5 = np.asarray([5,3,3,3,3,5,6,6.5,6.5,6,4.5,5])
    d6 = np.asarray([5,4,3.5,3.25,3,3,3,3,3.25,3.5,4,5])
    d7 = np.asarray([5,7,7,6.6,6.2,5.8,5.4,5,4.6,4.2,5,5])
    d8 = np.asarray([5,4,3.5,3.5,4,5,5,4,3.5,3.5,4,5])
    d9 = np.asarray([5,4,3.5,3.5,4,5,5,5,5,4.7,4.3,5])
    
    x = np.stack([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
    x -= x.mean(1,keepdims=True) # whiten
    x /= x.std(1,keepdims=True)
    x -= x[:,:1]  # signal starts and ends at 0
    
    templates = {'x': x/6., 't': np.linspace(-5, 5, len(d0))/6.,
                 'y': np.asarray([0,1,2,3,4,5,6,7,8,9])}
    return templates


# make a dataset
def make_dataset(args=None, template=None, ):
    templates = get_templates() if template is None else template
    args = get_dataset_args() if args is None else args
    set_seed(args.seed) # reproducibility
    
    xs, ys = [], []
    samples_per_class = args.num_samples // len(templates['y'])
    for label_ix in range(len(templates['y'])):
        for example_ix in range(samples_per_class):
            x = templates['x'][label_ix]
            t = templates['t']
            y = templates['y'][label_ix]
            x, new_t = transform(x, t, args) # new_t transformation is same each time
            xs.append(x) ; ys.append(y)
    
    batch_shuffle = np.random.permutation(len(ys)) # shuffle batch dimension
    xs = np.stack(xs)[batch_shuffle]
    ys = np.stack(ys)[batch_shuffle]
    
    if args.shuffle_seq: # maybe shuffle the spatial dimension
        seq_shuffle = np.random.permutation(args.final_seq_length)
        xs = xs[...,seq_shuffle]
    
    new_t = new_t/xs.std()
    xs = (xs-xs.mean())/xs.std() # center the dataset & set standard deviation to 1

    # train / test split
    split_ix = int(len(ys)*args.train_split)
    dataset = {'x': xs[:split_ix], 'x_test': xs[split_ix:],
               'y': ys[:split_ix], 'y_test': ys[split_ix:],
               't':new_t, 'templates': templates}
    return dataset



# we'll cache the dataset so that it doesn't have to be rebuild every time
# args must not be a dict
def get_dataset(args, path=None, verbose=True, download=True, regenerate=False, **kwargs):
    if 'args' in kwargs.keys() and kwargs['args'].shuffle_seq:
        shuffle = "_shuffle"
    else:
        shuffle = ""
    path = './mnist1d_data{}.pkl'.format(shuffle) if path is None else path

    assert not (download and regenerate), "You can either download the o.g. MNIST1D dataset or generate your own - but not both"
    try:
        if regenerate:
            raise ValueError("Regenerating dataset") # yes this is hacky
        if download:
            if os.path.exists(path):
                if verbose:
                    print("File already exists. Skipping download.")
            else:
                print("Downloading MNIST1D dataset from {}".format(args.url))
                r = requests.get(args.url, allow_redirects=True)
                open(path, 'wb').write(r.content)
                print("Saving to {}".format(path))
        dataset = from_pickle(path)
        if verbose:
            print("Successfully loaded data from {}".format(path))
    except:
        if verbose:
            print("Did or could not load data from {}. Rebuilding dataset...".format(path))
        dataset = make_dataset(args, **kwargs)
        to_pickle(dataset, path)
    return dataset

# transformations of the templates which will make them harder to classify
def pad(x, padding: tuple):
    """pad signal x with random number of zeros. Note, the signal is only padded at indices given by the interval in padding

    Parameters
    ----------
    x : _type_
        signal
    padding : tuple
        (low, high) corresponds to (start,end) of padding

    Returns
    -------
    _type_
        a padded signal
    """
    low, high = padding
    p = low + int(np.random.rand() * (high - low + 1))
    if len(x.shape) == 1:
        return np.concatenate([x, np.zeros((p))])
    else:
        padding = np.zeros((x.shape[0], p))
        return np.concatenate([x, padding], axis=-1)


def shear(x, scale=10):
    # TODO: add docstring
    coeff = scale * (np.random.rand() - 0.5)
    return x - coeff * np.linspace(-0.5, 0.5, len(x))


def translate(x, max_translation):
    # TODO: add docstring
    k = np.random.choice(max_translation)
    return np.concatenate([x[-k:], x[:-k]])


def corr_noise_like(x, scale):
    # TODO: add docstring
    noise = scale * np.random.randn(*x.shape)
    return gaussian_filter(noise, 2)


def iid_noise_like(x, scale):
    # TODO: add docstring
    noise = scale * np.random.randn(*x.shape)
    return noise


def interpolate(x, N):
    # TODO: add docstring
    scale = np.linspace(0, 1, len(x))
    new_scale = np.linspace(0, 1, N)
    new_x = interp1d(scale, x, axis=0, kind="linear")(new_scale)
    return new_x


def transform(x, y, args, eps=1e-8):
    new_x = pad(x + eps, args.padding)  # pad
    new_x = interpolate(new_x, args.template_len + args.padding[-1])  # dilate
    new_y = interpolate(y, args.template_len + args.padding[-1])
    new_x *= 1 + args.scale_coeff * (np.random.rand() - 0.5)  # scale
    new_x = translate(new_x, args.max_translation)  # translate

    # add noise
    mask = new_x != 0
    new_x = mask * new_x + (1 - mask) * corr_noise_like(new_x, args.corr_noise_scale)
    new_x = new_x + iid_noise_like(new_x, args.iid_noise_scale)

    # shear and interpolate
    new_x = shear(new_x, args.shear_scale)
    new_x = interpolate(new_x, args.final_seq_length)  # subsample
    new_y = interpolate(new_y, args.final_seq_length)
    return new_x, new_y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=3)


def from_pickle(path): # load something
    value = None
    with open(path, 'rb') as handle:
        value = pickle.load(handle)
    return value

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def plot_signals(xs, t, labels=None, args=None, ratio=2.6, do_transform=False, dark_mode=False, zoom=1):
    rows, cols = 1, 10
    fig = plt.figure(figsize=[cols*1.5,rows*1.5*ratio], dpi=60)
    for r in range(rows):
        for c in range(cols):
            ix = r*cols + c
            x, t = xs[ix], t
            ax = plt.subplot(rows,cols,ix+1)

            # plot the data
            if do_transform:
                assert args is not None, "Need an args object in order to do transforms"
                x, t = transform(x, t, args)  # optionally, transform the signal in some manner
            if dark_mode:
                plt.plot(x, t, 'wo', linewidth=6)
                ax.set_facecolor('k')
            else:
                plt.plot(x, t, 'k-', linewidth=2)
            if labels is not None:
                plt.title("label=" + str(labels[ix]), fontsize=22)

            plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)
            plt.gca().invert_yaxis() ; plt.xticks([], []), plt.yticks([], [])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout() ; plt.show()
    return fig

if __name__ == "__main__":
    dataset = MNIST1D_Dataset()
    x, y = dataset[0]

    print(x.shape, y.shape)
    print(y) 
    print(x)