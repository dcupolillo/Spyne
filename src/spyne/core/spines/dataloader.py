""" I/O operations for SpineDataset
    Created on January 21, 2026
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
import flammkuchen as fl
from tqdm import tqdm


class DataPaths:
    """
    Constants for data file paths.
    This class defines standard relative paths for various data files
    used in spine analysis.
    
    Assumes a base directory structure for organizing processed data
    and analysis results.
    """
    
    # Segmentation results (base paths — no algorithm suffix)
    SPINES_DATA = Path(r'processed/spines/spines_data.h5')
    DENDRITES_DATA = Path(r'processed/spines/dendrites_data.h5')
    SPINE_PREDICTIONS = Path(r'processed/spines/spine_predictions.h5')
    DENDRITE_PREDICTIONS = Path(r'processed/spines/dendrite_predictions.h5')
    SPINE_MASKS = Path(r'processed/spines/spine_masks.h5')
    DENDRITE_MASKS = Path(r'processed/spines/dendrite_masks.h5')

    # Imaging / folder paths
    PADDED_IMAGES_FOLDER = Path(r'processed/imaging/padded_images')
    CARE_DENOISED_IMAGES = Path(r'processed/imaging/care_denoised_images.h5')
    NNUNET_SPINE_PROBABILITY_FOLDER = Path(r'processed/spines/nnunet')
    NNUNET_DENDRITE_PROBABILITY_FOLDER = Path(r'processed/spines/nnunet')
    DEEPD3_SPINE_PROBABILITY_FOLDER = Path(r'processed/spines/deepd3')
    DEEPD3_DENDRITE_PROBABILITY_FOLDER = Path(r'processed/spines/deepd3')

    # Timeseries data (base paths — no algorithm suffix)
    ZSCORES_CA3 = Path(r'processed/imaging/zscores_CA3.h5')
    DFF_CA3 = Path(r'processed/imaging/dFF_CA3.h5')
    TS_CA3 = Path(r'processed/imaging/ts_CA3.h5')
    ZSCORES_BLA = Path(r'processed/imaging/zscores_BLA.h5')
    DFF_BLA = Path(r'processed/imaging/dFF_BLA.h5')
    TS_BLA = Path(r'processed/imaging/ts_BLA.h5')

    # Calcium events (base paths — no algorithm suffix)
    CALCIUM_EVENTS_BLA = Path(
        r'analysis/spines/calcium_event_probabilities_BLA.h5')
    CALCIUM_EVENTS_CA3 = Path(
        r'analysis/spines/calcium_event_probabilities_CA3.h5')


class SpineDataLoader:
    """
    Handles I/O operations for spine analysis data.
    Useful for loading presaved .h5 files and saving analysis results.
    
    This class manages:
    - Loading and saving .h5 files
    - File path resolution
    - Data validation during I/O
    - Batch loading operations
    """
    
    def __init__(
            self,
            base_path: str | Path
    ) -> None:
        """
        Initialize data loader.
        
        Parameters
        ----------
        base_path : str | Path
            Base directory path for data files.
        """
        self.base_path = Path(base_path)
        self._files_mapping = self._create_files_mapping()
    
    def _create_files_mapping(self) -> dict:
        """
        Create mapping of attribute names to file paths.
        
        Returns
        -------
        dict
            Dictionary mapping attribute names to file paths.
        """
        b = self.base_path
        mapping = {
            # Base segmentation paths (no algorithm or denoising suffix)
            'spines_data': b / DataPaths.SPINES_DATA,
            'dendrites_data': b / DataPaths.DENDRITES_DATA,
            'spine_predictions': b / DataPaths.SPINE_PREDICTIONS,
            'dendrite_predictions': b / DataPaths.DENDRITE_PREDICTIONS,
            'spine_masks': b / DataPaths.SPINE_MASKS,
            'dendrites_masks': b / DataPaths.DENDRITE_MASKS,
            'combined_spines_masks': b / DataPaths.SPINE_MASKS,

            # Folders and special files (never receive variant suffixes)
            'padded_images_folder': b / DataPaths.PADDED_IMAGES_FOLDER,
            'care_denoised_images': b / DataPaths.CARE_DENOISED_IMAGES,
            'deepd3_probability_folder': b / DataPaths.DEEPD3_SPINE_PROBABILITY_FOLDER,
            'deepd3_prediction_folder': b / DataPaths.DEEPD3_SPINE_PROBABILITY_FOLDER,
            'nnunet_probability_folder': b / DataPaths.NNUNET_SPINE_PROBABILITY_FOLDER,
            'nnunet_prediction_folder': b / DataPaths.NNUNET_SPINE_PROBABILITY_FOLDER,

            # Base timeseries (no algorithm suffix, legacy)
            'zscores_CA3': b / DataPaths.ZSCORES_CA3,
            'dFF_CA3': b / DataPaths.DFF_CA3,
            'ts_CA3': b / DataPaths.TS_CA3,
            'zscores_BLA': b / DataPaths.ZSCORES_BLA,
            'dFF_BLA': b / DataPaths.DFF_BLA,
            'ts_BLA': b / DataPaths.TS_BLA,

            # Base calcium events (no algorithm suffix, legacy)
            'calcium_events_probabilities_BLA': b / DataPaths.CALCIUM_EVENTS_BLA,
            'calcium_events_probabilities_CA3': b / DataPaths.CALCIUM_EVENTS_CA3,
        }

        # Keys that should never receive algorithm or denoising suffixes.
        _fixed_keys = {
            'combined_spines_masks', 'care_denoised_images',
            'padded_images_folder',
            'deepd3_probability_folder', 'deepd3_prediction_folder',
            'nnunet_probability_folder', 'nnunet_prediction_folder',
        }

        # Supported segmentation algorithms. To add a new one (e.g. cellpose),
        # append its suffix here.
        supported_algo_suffixes = ('_deepd3', '_nnunet')

        # Active denoising backends. To add a new one (e.g. Noise2Void),
        # append its suffix here.
        active_denoise_suffixes = ('_care',)

        # Step 1 — generate algorithm-specific variants from base .h5 paths.
        for algo_suffix in supported_algo_suffixes:
            algo_variants = {
                f"{key}{algo_suffix}": (
                    path.parent / f"{path.stem}{algo_suffix}{path.suffix}"
                )
                for key, path in list(mapping.items())
                if path.suffix == '.h5'
                and key not in _fixed_keys
            }
            mapping.update(algo_variants)

        # Step 2 — generate denoising variants from algorithm-specific paths.
        for denoise_suffix in active_denoise_suffixes:
            denoised_variants = {
                f"{key}{denoise_suffix}": (
                    path.parent / f"{path.stem}{denoise_suffix}{path.suffix}"
                )
                for key, path in list(mapping.items())
                if path.suffix == '.h5'
                and any(key.endswith(s) for s in supported_algo_suffixes)
                and not any(key.endswith(d) for d in active_denoise_suffixes)
            }
            mapping.update(denoised_variants)

        return mapping

    def load_file(
        self,
        filepath: str | Path,
    ) -> tuple:
        """
        Load a single .h5 file and return attribute name and data.
        
        Parameters
        ----------
        filepath : str | Path
            Path to the .h5 file to load.
            
        Returns
        -------
        tuple
            Tuple containing (attribute_name, loaded_data).
            
        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If file extension validation fails.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist")
        
        if filepath.suffix != '.h5':
            raise ValueError(f"File {filepath} must be a .h5 file")
        
        # Determine attribute name from filename mapping
        filename = filepath.name
        attribute_name = self._get_attribute_name(filename)
        
        try:
            data = fl.load(filepath)
            return attribute_name, data
        
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return attribute_name, []
    
    def _get_attribute_name(self, filename: str) -> str:
        """
        Get attribute name from filename using mapping.
        
        Parameters
        ----------
        filename : str
            Name of the file.
            
        Returns
        -------
        str
            Attribute name for the file.
        """
        for attr, mapped_filename in self._files_mapping.items():
            if filename == Path(mapped_filename).name:
                return attr
        
        # If not in mapping, use stem of filename
        return Path(filename).stem
    
    def load_multiple_files(
        self,
        file_paths: list = None
    ) -> dict:
        """
        Load multiple .h5 files and return as dictionary.
        
        Parameters
        ----------
        file_paths : list, optional
            List of file paths to load. If None, attempts to load all
            mapped files that exist.
            
        Returns
        -------
        dict
            Dictionary mapping attribute names to loaded data.
        """
        if file_paths is None:

            file_paths = [
                full_path
                for full_path in self._files_mapping.values()
                if full_path.exists() and full_path.is_file()
            ]
        
        loaded_data = {}
        
        if file_paths:
            for filepath in tqdm(
                file_paths,
                desc="Loading .h5 data",
                total=len(file_paths)
            ):
                attr_name, data = self.load_file(filepath)
                loaded_data[attr_name] = data
        
        # Initialize missing attributes as empty lists
        for attr in self._files_mapping.keys():
            if attr not in loaded_data:
                loaded_data[attr] = []
        
        return loaded_data
    
    def save_file(
        self,
        data: any,
        output_path: str | Path,
    ) -> None:
        """
        Save data to an .h5 file.
        
        Parameters
        ----------
        data : Any
            Data to save.
        output_path : str | Path
            Name of the file will be saved.
            
        Raises
        ------
        ValueError
            If filename doesn't end with .h5 or data is empty.
        FileNotFoundError
            If output directory doesn't exist and create_dirs is False.
        NotADirectoryError
            If output_path exists but is not a directory.
        """
        output_path = Path(output_path)
        
        if not output_path.name.endswith('.h5'):
            raise ValueError(f"Filename {output_path.name} must end with '.h5'")
        
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            print(f"Warning: No data to save for {output_path.name}")
            return
        
        # Create directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        if data is not None and len(data) > 0:
            fl.save(output_path, data)
            print(f"Saved {output_path.name} to {output_path.parent}")
    
    def save_multiple_files(
        self,
        data_dict: dict,
        output_path: str | Path,
        filename_mapping: dict = None
    ) -> None:
        """
        Save multiple datasets to .h5 files.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping attribute names to data.
        output_path : str | Path
            Base directory where files will be saved.
        filename_mapping : dict, optional
            Mapping from attribute names to relative file paths.
            If None, uses default mapping with full relative paths.
        """
        if filename_mapping is None:
            # Use full relative paths from _files_mapping
            filename_mapping = {
                attr: path 
                for attr, path in self._files_mapping.items()
            }
        
        output_path = Path(output_path)
        
        for attr_name, data in data_dict.items():
            if attr_name in filename_mapping:
                relative_path = Path(filename_mapping[attr_name])
                full_path = output_path / relative_path
                
                # Create parent directories if needed
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.save_file(data, full_path)
    
    def get_existing_files(self) -> dict:
        """
        Get dictionary of existing data files.
        
        Returns
        -------
        dict
            Dictionary mapping attribute names to existing file paths.
        """
        return {
            attr: full_path
            for attr, full_path in self._files_mapping.items()
            if full_path.exists()
        }
    
    def get_missing_files(self) -> dict:
        """
        Get dictionary of missing data files.
        
        Returns
        -------
        dict
            Dictionary mapping attribute names to expected file paths.
        """
        return {
            attr: full_path
            for attr, full_path in self._files_mapping.items()
            if not full_path.exists()
        }