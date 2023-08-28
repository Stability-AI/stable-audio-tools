from prefigure.prefigure import get_all_args
import json
import torch
import numpy as np
from harmonai_tools.models.pca import PCA

from harmonai_tools.data.dataset import create_dataloader_from_configs_and_args
from harmonai_tools.models import create_model_from_config
from harmonai_tools.training.utils import copy_state_dict


def main():

    args = get_all_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    model_type = model_config["model_type"]

    assert model_type == "autoencoder", "Only autoencoder models are supported for PCA analysis"

    train_dl = create_dataloader_from_configs_and_args(model_config, args, dataset_config)

    model = create_model_from_config(model_config)

    model.to(device)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, torch.load(args.pretrained_ckpt_path)["state_dict"])
    
    if args.pretransform_ckpt_path:
        print("Loading pretransform from checkpoint")
        model.pretransform.load_state_dict(torch.load(args.pretransform_ckpt_path)["state_dict"])
        print("Done loading pretransform from checkpoint")

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})

    latents = []

    max_latents = 1000

    for i, batch in enumerate(train_dl):
        print(f"Batch {i}")
        reals, _ = batch

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        reals = reals.to(device)
        sample_latents = model.encode(reals).detach().cpu()
        latents.append(sample_latents)
        if len(latents) > max_latents:
            break

    latents = torch.cat(latents, dim=0)

    batch_size, latent_dim, sequence_length = latents.shape

    latents = latents.reshape(-1, latent_dim)

    print(latents.shape)

    pca = PCA(n_components=latent_dim)
    pca.fit(latents)
    # pca_data = pca.transform(latents)
    # components = pca.components_
    explained_variance_ratios = pca.explained_variance_ratio_.numpy()
    cumulative_variance_ratio = np.cumsum(explained_variance_ratios)

    for fidelity in [0.8, 0.9, 0.95, 0.99]:

        informative_dimensions = np.argmax(cumulative_variance_ratio >= fidelity) + 1

        print(f"informative dimensions with fidelity {fidelity}: {informative_dimensions}")

    model.latent_pca = pca

    new_ckpt = {
        'state_dict': model.state_dict()
    }

    torch.save(new_ckpt, './model_pca.pt')

if __name__ == '__main__':
    main()