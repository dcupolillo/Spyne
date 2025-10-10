
from pathlib import Path
import flammkuchen as fl

def save_metadata(
    metadata: dict,
    save_path: str or Path,
    overwrite: bool = False
) -> None:
    """
    Save the metadata to a .json file for fast loading and inspection.

    Parameters
    ----------
    save_path : Path, optional
        Path where to save the metadata. If None, uses 'metadata.json' in the dataset folder.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if metadata is None:
        raise ValueError("No metadata to save. Load metadata first.")

    save_path = Path("metadata.json") if save_path is None else Path(save_path)

    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists. Use overwrite=True to overwrite.")


    # Save metadata using flammkuchen (HDF5)
    fl.save(save_path, metadata)
    file_size_mb = save_path.stat().st_size / (1024**2)
    
    print(f"✓ Saved metadata ({file_size_mb:.2f} MB) to {save_path}.")
    
    return
