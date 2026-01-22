"""Data loading and I/O operations for imaging datasets.

Created on January 21, 2026
@author: dcupolillo
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import flammkuchen as fl
from spyne.core.imaging.load import (
    load_metadata_from_tiff, load_imaging_data_from_tiff)


class ImagingDataPaths:
    """
    Constants for data file paths.
    This class defines standard relative paths for various data files
    used in spine analysis.
    
    Assumes a base directory structure for organizing processed data
    and analysis results.
    """

    RAW_FOLDER = Path(r"raw")
    RAW_TIFF_FOLDER = RAW_FOLDER / "tiff"
    RAW_ABF_FOLDER = RAW_FOLDER / "abf"

    PREPROCESSED_DATA = Path(r"processed/imaging/processed_imaging.h5")
    METADATA = Path(r"processed/imaging/metadata.h5")
    


class ImagingDataLoader:
    """    
    Handles all I/O operations including file discovery, metadata extraction,
    data loading from various formats, and saving processed results.
    Supports lazy loading and memory optimization strategies.
    """
    
    def __init__(
            self,
            folder: Path
    ) -> None:
        """
        Initialize data loader.
        
        Parameters
        ----------
        folder : Path
            Base folder containing imaging data.
        """
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid dataset folder: {folder}")
        
        self.folder = folder
        
        # Initialize file lists
        self._discover_files()
        self.files_mapping = self._create_files_mapping()

    def _create_files_mapping(self) -> dict:
        """
        Create mapping of attribute names to file paths.
        
        Returns
        -------
        dict
            Dictionary mapping attribute names to file paths.
        """

        return {
            'preprocessed_data': self.folder / ImagingDataPaths.PREPROCESSED_DATA,
            'metadata': self.folder / ImagingDataPaths.METADATA,
            'swc_file': self.swc_file,
            'stack_file': self.stack_file,
            'tiff_files': self.file_list,
            'abf_files': self.abf_file_list,
        }
        
    def _discover_files(self) -> None:
        """
        Discover all relevant files in the dataset folder.
        Populates lists of TIFF and ABF files, as well as stack and SWC files.

        Raises
        ------
        Exception
            If no imaging or ABF files are found.
        
        """
        
        raw_tiff_folder = self.folder / ImagingDataPaths.RAW_TIFF_FOLDER
        raw_abf_folder = self.folder / ImagingDataPaths.RAW_ABF_FOLDER
        raw_folder = self.folder / ImagingDataPaths.RAW_FOLDER
        
        # Find TIFF files
        self.file_list = [
            file_path for file_path in raw_tiff_folder.rglob('*')
            if file_path.suffix.lower() in [".tif", ".tiff"]
        ]
        
        # Find ABF files
        self.abf_file_list = [
            file_path for file_path in raw_abf_folder.rglob('*')
            if file_path.suffix.lower() in [".abf"]
        ]
        
        # Find stack and SWC files
        self.stack_file = self._find_stack_file(raw_folder)
        self.swc_file = self._find_swc_file(raw_folder)
        
        if not self.file_list:
            raise Exception(f"No imaging files found in {raw_tiff_folder}.")
        
        if not self.abf_file_list:
            raise Exception(f"No ABF files found in {raw_abf_folder}.")
    
    def _find_stack_file(self, folder: Path) -> Path:
        """
        Find stack file in the raw folder.
        Does not assume a specific filename, but location is crucial.
        """
        stack_files = [
            f for f in folder.glob("*.tif*")
            if 'stack' in f.stem.lower()
        ]
        return stack_files[0] if stack_files else None
    
    def _find_swc_file(self, folder: Path) -> Path:
        """
        Find SWC file in the raw folder.
        Does not assume a specific filename, but location is crucial.
        """
        swc_files = list(folder.glob('*.swc'))
        return swc_files[0] if swc_files else None
    
    def load_metadata(
        self,
        dataset_instance,
        filepath: str or Path = None,
    ) -> tuple:
        """
        Load metadata for all ROIs.
        
        Parameters
        ----------
        dataset_instance
            ImagingDataset instance (needed for compatibility with existing functions).
        filepath : str or Path
            Filename for saved metadata. Default is "metadata.h5". 
            If None, attempts to load from default location.
            
        Returns
        -------
        tuple
            (metadata_list, n_rois, roi_list)

        Notes
        -----
        First attempts to load from saved .h5 file.
        If that fails, falls back to loading from raw TIFF files.
        """
        if filepath is None:
            filepath = self.files_mapping['metadata']
        
        if filepath.suffix != '.h5':
            raise ValueError(f"File {filepath} must be a .h5 file")
        
        filepath = Path(filepath)

        # Try to load from saved file first
        if filepath.exists():
            try:
                saved_data = fl.load(filepath)
                
                # Handle both old and new format
                if isinstance(saved_data, dict) and 'metadata' in saved_data:
                    metadata = saved_data['metadata']  # New format
                else:
                    metadata = saved_data  # Old format
                
                n_rois = len(metadata)
                roi_list = np.arange(n_rois)
                
                print(f"Loaded metadata from {filepath}")
                return metadata, n_rois, roi_list
                
            except Exception as e:
                print(f"Failed to load saved metadata: {e}. Loading from raw files...")
        
        # Fall back to loading from raw TIFF files
        print("Loading metadata from raw TIFF files...")
        n_rois, metadata = load_metadata_from_tiff(dataset_instance)
        roi_list = np.arange(n_rois)
        
        return metadata, n_rois, roi_list
    
    def load_imaging_data(
        self,
        dataset_instance,
        filepath: str or Path = None
    ) -> list:
        """
        Load and process imaging data.
        
        Parameters
        ----------
        dataset_instance
            ImagingDataset instance (needed for compatibility).
        filepath : str or Path, optional
            Filepath for processed data. Default is "processed_imaging.h5". 
            If None, attempts to load from default location.
            
        Returns
        -------
        list
            List of processed imaging data arrays.
        """
        if filepath is None:
            filepath = self.files_mapping['preprocessed_data']

        if filepath.suffix != '.h5':
            raise ValueError(f"File {filepath} must be a .h5 file")

        filepath = Path(filepath)

        if filepath.exists():
            try:
                preprocessed_data = fl.load(filepath)

                if isinstance(preprocessed_data, dict) and 'processed_arrays' in preprocessed_data:
                    data = preprocessed_data['processed_arrays']  # Correct key
                else:
                    data = preprocessed_data  # Old format (direct arrays)

                # Validate data
                if not data:
                    raise ValueError("No arrays found in saved file")

                print(f"Loaded processed data from {filepath}")
                return data
                
            except Exception as e:
                print(f"Failed to load processed data: {e}. Processing from raw files...")
        
        # Fall back to loading from raw TIFF files
        print("Processing imaging data from raw TIFF files...")
        data = load_imaging_data_from_tiff(dataset_instance)

        return data
    
    def save_metadata(
        self,
        metadata: list,
        output_path: str or Path,
    ) -> Path:
        """
        Save metadata to file.
        
        Parameters
        ----------
        metadata : list
            List of metadata dictionaries.
        output_path : str or Path
            Output file path.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
            
        Returns
        -------
        Path
            Path to saved file.
        """
        if output_path is None:
            output_path = self.files_mapping['metadata']
        
        output_path = Path(output_path)

        if not output_path.name.endswith('.h5'):
            raise ValueError(f"Filename {output_path.name} must end with '.h5'")

        # Create folder if does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata using flammkuchen (HDF5)
        if metadata is not None and len(metadata) > 0:
            fl.save(output_path, metadata)
            print(f"Metadata saved to {output_path}")
    
    def save_processed_data(
        self,
        data: list,
        processing_params: dict,
        output_path: str or Path,
        overwrite: bool = False
    ) -> Path:
        """
        Save processed imaging data.
        
        Parameters
        ----------
        data : list
            List of processed data arrays.
        processing_params : dict
            Parameters used for processing.
        output_path : str or Path, optional
            Output file path. If None, uses default location.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
            
        Returns
        -------
        Path
            Path to saved file.
        """
        if output_path is None:
            output_path = self.files_mapping['preprocessed_data']
            
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.exists() and not overwrite:
            raise FileExistsError(f"File exists: {save_path}. Use overwrite=True")
        
        if not data:
            raise ValueError("Data list is empty - nothing to save")
        
        # Simple data structure - just arrays and processing params
        save_data = {
            'processed_arrays': data,
            'processing_params': processing_params
        }
        
        fl.save(save_path, save_data)
        file_size_mb = save_path.stat().st_size / (1024**2)
        print(f"✓ Saved processed data ({file_size_mb:.1f} MB) to {save_path}")
        
        return save_path
    
    def check_processed_data_exists(self) -> bool:
        """
        Check if processed data file exists.
        
        Parameters
        ----------
        filename : str, optional
            Filename to check. Default is "processed_imaging.h5".
            
        Returns
        -------
        bool
            True if file exists, False otherwise.
        """
        return ImagingDataPaths.PREPROCESSED_DATA.exists()
    
    def check_metadata_exists(self) -> bool:
        """
        Check if metadata file exists.
        
        Parameters
        ----------
        filename : str, optional
            Filename to check. Default is "metadata.h5".
            
        Returns
        -------
        bool
            True if file exists, False otherwise.
        """
        return ImagingDataPaths.METADATA.exists()