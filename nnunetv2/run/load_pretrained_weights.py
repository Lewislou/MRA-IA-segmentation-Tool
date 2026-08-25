import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def load_pretrained_weights(network, fname, verbose=False):
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']
    is_ddp = isinstance(network, DDP)

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    model_dict = network.state_dict()

    for key, _ in model_dict.items():
        if is_ddp:
            key_pretrained = key[7:]
        else:
            key_pretrained = key
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key_pretrained in pretrained_dict,\
                f"Key {key_pretrained} is missing in the pretrained model weights. The pretrained weights do not seem to be "\
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key_pretrained].shape,\
                f"The shape of the parameters of key {key_pretrained} is not the same. Pretrained model: "\
                f"{pretrained_dict[key_pretrained].shape}; your network: {model_dict[key]}. The pretrained model "\
                f"does not seem to be compatible with your network."

    pretrained_dict = {'module.' + k if is_ddp else k: v
                       for k, v in pretrained_dict.items()
                       if (('module.' + k if is_ddp else k) in model_dict) and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, _ in pretrained_dict.items():
            print(key[7:] if is_ddp else key)
        print("################### Done ###################")
    network.load_state_dict(model_dict)
