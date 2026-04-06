import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from typing import Iterator

class HyperSpectralDataset(Dataset):
    """
    Defining this class so that we can keep track of spectra, abundances, names, and indices.

    Args:
        save_path (str): The path csv was saved to
        spec_range (list[int]): An inclusive double entry list of spec range ints
        all_spectra (bool): Whether to include all spectra or only a subset
        start_idx (int): The starting index for selecting spectra
    """
    def __init__(self, save_path:(str), spec_range:(list[int]), all_spectra:(bool)=True, start_idx:(int)=24):

        spectra, abundances, names, indices = self._vals_from_csv(save_path, spec_range)

        if not all_spectra:
            spectra = spectra[start_idx:]
            abundances = abundances[start_idx:]
            names = names[start_idx:]
            indices = indices[start_idx:]

        self.spectra = spectra
        self.abundances = abundances
        self.names = names
        self.indices = indices

    def _vals_from_csv(self, save_path:(str), spec_range:(list[int])) -> torch.Tensor | torch.Tensor | list[str] | list[int]:
        """
        Gets the vals from a csv file that was exported via dataset creation in the training of unmixing models.

        Args:
            save_path (str): The path csv was saved to
            spec_range (list[int]): An inclusive double entry list of spec range ints
        
        Returns:
            spectra_tensor (torch.Tensor): Tensor of all the specra in csv
            abundances_tensor (torch.Tensor): Tensor of all the abundances in csv
            names (list[str]): List of all string spectral names
            indices (list[int]): List of all true spectral integer indices
        """
        df = pd.read_csv(save_path)

        spectral_cols = [str(w) for w in range(spec_range[0], spec_range[1]+1, 10)]
        spectra = df[spectral_cols].values.astype("float32")
        spectra_tensor = torch.from_numpy(spectra)

        abundances = df[["gv_fraction","npv_fraction","soil_fraction"]].values.astype("float32")
        abundances_tensor = torch.from_numpy(abundances)

        # Get the spectral names
        names = df['Spectra'].to_list()

        # Get the original indices
        indices = list(range(len(names)))
        
        return spectra_tensor, abundances_tensor, names, indices

    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, idx):

        return {
            'spectrum': self.spectra[idx],
            'abundances': self.abundances[idx],
            'names': self.names[idx],
            'orig_index': self.indices[idx]
        }

def get_inf_iterator(dataloader:(DataLoader)) -> Iterator:
    """
    Gets the infinite iterator, it can be infinitely iterated through.

    Args:
        dataloader (DataLoader): An already prepared dataloader
    
    Returns:
        cycle(dataloader): An infinitely iterable dataloader
    """
    def cycle(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    return cycle(dataloader)

def get_data(save_path:(str), spec_range:(list[int]), seed:(int)=3169, n_train:(int)=1500, n_val:(int)=100, n_test:(int)=123, batch_size:(int)=8, num_workers:(int)=4, prefetch_factor:(int)=20, all_spectra:(bool)=True, start_idx:(int)=24, SNV:(bool)=False) -> list[Iterator, Iterator, Iterator]:

    ds = HyperSpectralDataset(save_path, spec_range, all_spectra, start_idx)

    # Split indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=generator)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    # Apply statnorm, using only training data stats 
    if not SNV:
        train_spectra = ds.spectra[train_idx]
        train_mean = train_spectra.mean(dim=0, keepdim=True)
        train_std = train_spectra.std(dim=0, keepdim=True)
        ds.spectra = (ds.spectra - train_mean) / torch.clamp(train_std, min=1e-8)
    if SNV:
        means = ds.spectra.mean(dim=1, keepdim=True)
        stds = ds.spectra.std(dim=1, keepdim=True)
        ds.spectra = (ds.spectra - means) / torch.clamp(stds, min=1e-8)

    # Create sub-datasets
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)
    ds_test = Subset(ds, test_idx)

    # Create dataloaders
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=True 
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=n_val,
        shuffle=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=n_test,
        shuffle=False,
    )

    return [get_inf_iterator(dl_train), get_inf_iterator(dl_val), iter(dl_test)]