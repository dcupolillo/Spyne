from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import tifffile
import flammkuchen as fl

from spyne.core.spines.analysis.backends.base_backend import (
    BaseBackend, InferenceJob, InferenceResult)
from spyne.core.spines.analysis.padding import unpad_predictions


# ---------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------

@dataclass
class CSBDeepJob(InferenceJob):
    """
    Input specification for CSBDeep (CARE) denoising.

    CSBDeep performs content-aware image restoration to denoise images
    before segmentation, improving segmentation quality.
    """
    axes: str = "YX"  # Image axes configuration (e.g., "YX" for 2D, "ZYX" for 3D)
    original_dimensions: tuple[int, ...] | list[int] = field(
        default_factory=list
    )


@dataclass
class CSBDeepResult(InferenceResult):
    """
    Normalized result returned by the CSBDeep backend.

    Extends InferenceResult with CSBDeep-specific outputs:
    - denoised_images: Array of denoised images ready for segmentation

    All other fields inherited from InferenceResult.
    """
    denoised_images: np.ndarray | None = None


# ---------------------------------------------------------------------
# CSBDeep backend
# ---------------------------------------------------------------------

class CSBDeepBackend(BaseBackend):
    """
    Backend wrapper for CSBDeep (CARE) image denoising.

    This class encapsulates:
    - command construction for CSBDeep CLI
    - subprocess execution in isolated environment
    - loading denoised images
    - returning processed images for segmentation pipeline

    Notes
    -----
    CSBDeep uses content-aware image restoration (CARE) to denoise
    images before segmentation, which can significantly improve
    segmentation quality especially for low SNR data.
    
    The exact CLI command structure will be determined based on the
    CSBDeep repository with cli implementation at:
    https://github.com/dcupolillo/CSBDeep
    forked from:
    https://github.com/CSBDeep/CSBDeep
    """

    name = "csbdeep"

    def __init__(self, python_executable: Path | str) -> None:
        """
        Parameters
        ----------
        python_executable
            Python interpreter inside the environment where CSBDeep is
            installed.
        """
        self.python_executable = Path(python_executable).resolve()
        self._temp_dir = None  # Track temporary directory for cleanup

    def run(self, job: CSBDeepJob) -> CSBDeepResult:
        """
        Run the full CSBDeep denoising workflow.

        This method is the public entry point used by the rest of the
        pipeline. It executes denoising, unpads to original dimensions,
        and returns results (saving is handled by the calling pipeline).

        Parameters
        ----------
        job
            CSBDeep job specification with all necessary parameters.

        Returns
        -------
        CSBDeepResult
            Result containing denoised images (unpadded) and metadata.
        """
        # Normalize and validate paths
        job = job._normalize_job_paths()
        self._validate_job(job)

        # Prepare output directory
        self._prepare_output_folder(job.output_dir)

        # Build command
        cmd = self._build_command(job)

        # Execute CSBDeep
        start_time = time.perf_counter()
        completed = self._run_subprocess(cmd)
        runtime_s = time.perf_counter() - start_time

        # Load denoised images (still padded)
        denoised_images = self._load_denoised_images(job)

        # Unpad to original dimensions
        if job.original_dimensions is not None:
            denoised_images = unpad_predictions(
                denoised_images,
                job.original_dimensions
            )

        # Package results
        return CSBDeepResult(
            backend_name=self.name,
            output_path=job.output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=cmd,
            return_code=completed.returncode,
            runtime_s=runtime_s,
            metadata={
                "model_name": str(job.model_name),
                "axes": job.axes,
                "n_images": len(denoised_images),
            },
            denoised_images=denoised_images,
        )

    def _build_command(self, job: CSBDeepJob) -> list[str]:
        """
        Build the CSBDeep subprocess command.

        TODO: Update this once CSBDeep CLI structure is finalized.
        Current placeholder assumes a structure similar to:
        python -m csbdeep.predict --input <dir> --output <dir> --model <path>

        Parameters
        ----------
        job
            CSBDeep job specification with all necessary parameters.

        Returns
        -------
        list[str]
            Command to execute CSBDeep denoising.
        """
        # Verify python executable exists
        python_path = Path(self.python_executable)
        if not python_path.exists():
            raise FileNotFoundError(
                f"Python executable not found: {python_path}\n\n"
                f"Please verify:\n"
                f"1. The CSBDeep conda environment exists:\n"
                f"   conda env list\n"
                f"2. The path in spine_config.yaml 'care_python_executable' "
                f"is correct:\n"
                f"   Check: src/spyne/config/spine_config.yaml\n"
                f"3. If using a different environment name or location, "
                f"update the config file accordingly.\n\n"
                f"Expected path: {python_path}"
            )
        
        # Placeholder command structure - update based on actual CSBDeep CLI
        cmd = [
            str(self.python_executable),
            "-m", "csbdeep.scripts.cli_predict",
            "--input", str(job.input_path),
            "--output", str(job.output_dir),
            "--model", str(job.model_name.parent),
            "--axes", job.axes,
        ]
        return cmd

    def _parse_result(
        self,
        job: InferenceJob,
        command: list[str],
        return_code: int,
        stdout: str,
        stderr: str,
        runtime_s: float,
    ) -> InferenceResult:
        """
        Parse subprocess results (not used - run() is overridden).

        CSBDeep requires custom loading of denoised images.
        """
        return InferenceResult(
            backend_name=self.name,
            output_path=job.output_dir,
            stdout=stdout,
            stderr=stderr,
            command=command,
            return_code=return_code,
            runtime_s=runtime_s,
            metadata={},
        )

    def _load_denoised_images(self, job: CSBDeepJob) -> np.ndarray:
        """
        Load denoised images from CSBDeep output.

        CSBDeep should write denoised images to the output directory,
        preserving the input filenames.

        Parameters
        ----------
        job
            CSBDeep job specification containing output directory path.

        Returns
        -------
        np.ndarray
            Array of denoised images with shape (n_images, height, width).

        Raises
        ------
        FileNotFoundError
            If expected output files are not found.
        """
        # Get input image files to match output filenames
        input_images = sorted(job.input_path.glob('*.tif'))
        n_images = len(input_images)

        if n_images == 0:
            raise FileNotFoundError(
                f"No input images found in {job.input_path}"
            )

        denoised_images = [None] * n_images

        # Load denoised images matching input filenames
        for idx, img_file in enumerate(input_images):
            # Assume CSBDeep preserves input filenames
            # Update this based on actual CSBDeep output naming convention
            output_path = job.output_dir / img_file.name

            if not output_path.exists():
                raise FileNotFoundError(
                    f"Denoised image not found: {output_path}"
                )

            denoised_images[idx] = tifffile.imread(output_path)

        return np.array(denoised_images)

    @staticmethod
    def save_as_h5(
        images: np.ndarray,
        output_path: Path | str
    ) -> None:
        """
        Save denoised images as h5 file.

        Parameters
        ----------
        images
            Array of denoised images (N, H, W) or (N, H, W, C).
        output_path
            Path for output h5 file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save using flammkuchen (consistent with project conventions)
        fl.save(str(output_path), {'denoised_images': images})

        print(f"Saved denoised images to {output_path}")

    @staticmethod
    def load_denoised_h5(h5_path: Path | str) -> np.ndarray:
        """
        Load denoised images from h5 file.

        Utility method to load previously saved denoised images without
        re-running the full denoising pipeline.

        Parameters
        ----------
        h5_path
            Path to h5 file containing denoised images.

        Returns
        -------
        np.ndarray
            Array of denoised images.

        Raises
        ------
        FileNotFoundError
            If h5 file doesn't exist.
        KeyError
            If 'denoised_images' key not found in h5 file.

        Examples
        --------
        >>> backend = CSBDeepBackend(python_executable="...")
        >>> images = backend.load_denoised_h5(
        ...     "processed/imaging/care_denoised_images.h5"
        ... )
        """
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(
                f"Denoised images h5 file not found: {h5_path}"
            )

        data = fl.load(str(h5_path))
        if 'denoised_images' not in data:
            raise KeyError(
                f"'denoised_images' key not found in {h5_path}"
            )

        raw = data['denoised_images']
        if isinstance(raw, dict):
            # flammkuchen stored ragged arrays as numbered sub-groups
            return [raw[str(i)] for i in range(len(raw))]
        return raw
