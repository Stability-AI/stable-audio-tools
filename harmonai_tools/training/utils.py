import torch

def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            print("Loading key", key)
            model_state_dict[key] = state_dict[key]
        
    #model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict, strict=False)