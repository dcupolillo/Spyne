from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import tifffile

from spyne.core.spines.analysis.backends.base_backend import (
    BaseBackend, InferenceJob, InferenceResult)
from spyne.core.spines.analysis.padding import unpad_predictions

# ---------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------

@dataclass
class DeepD3Job(InferenceJob):
    """
    Input specification for one DeepD3 batch inference run.

    This is backend-specific rather than fully generic because DeepD3 has
    its own assumptions:
    - input is a folder of padded TIFF images
    - output is a folder containing 'spines' and 'dendrites' subfolders
    - unpadding is required after prediction
    """
    original_dimensions: tuple[int, ...] | list[int] = field(
        default_factory=list
    )
    device: str = "cpu"


@dataclass
class DeepD3Result(InferenceResult):
    """
    Normalized result returned by the DeepD3 backend.

    Extends InferenceResult with DeepD3-specific outputs:
    - spine_probabilities: probability maps for detected spines
    - dendrite_probabilities: probability maps for detected dendrites

    All other fields (backend_name, success, output_path, stdout, stderr,
    command, return_code, runtime_s, metadata) are inherited from
    InferenceResult.
    """
    spine_probabilities: np.ndarray | None = None
    dendrite_probabilities: np.ndarray | None = None


# ---------------------------------------------------------------------
# DeepD3 backend
# ---------------------------------------------------------------------

class DeepD3Backend(BaseBackend):
    """
    Backend wrapper for DeepD3 batch inference.

    This class encapsulates:
    - command construction
    - subprocess execution
    - loading DeepD3 outputs
    - unpadding predictions
    - overwriting saved outputs with unpadded images

    Notes
    -----
    DeepD3 is a deep learning-based tool for spine and dendrite detection
    in microscopy images. 
    The exact CLI command structure will be determined based on the
    DeepD3 repository with cli implementation at:
    https://github.com/dcupolillo/DeepD3
    forked from:
    https://github.com/ankilab/DeepD3
    """

    name = "deepd3"

    def __init__(self, python_executable: Path | str) -> None:
        """
        Parameters
        ----------
        python_executable
            Python interpreter inside the environment where DeepD3 is installed.
        """
        self.python_executable = Path(python_executable).resolve()

    def run(self, job: DeepD3Job) -> DeepD3Result:
        """
        Run the full DeepD3 inference workflow.

        This method is the public entry point used by the rest of the pipeline.
        """
        # Normalize and validate all paths before doing anything else.
        job = job._normalize_job_paths()
        self._validate_job(job)

        # Make sure the output folder starts clean, as in your current function.
        self._prepare_output_folder(job.output_dir)

        # Build the exact subprocess command for this backend.
        cmd = self._build_command(job)

        # Execute DeepD3 in its own environment/process.
        start_time = time.perf_counter()
        completed = self._run_subprocess(cmd)
        runtime_s = time.perf_counter() - start_time

        # Check if DeepD3 succeeded before trying to load predictions
        if completed.returncode != 0:
            raise RuntimeError(
                f"DeepD3 inference failed with return code "
                f"{completed.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        # Load the padded probability maps produced by DeepD3.
        spine_preds, dendrite_preds = self._load_predictions(job)

        # Convert padded predictions back to the original image dimensions.
        spine_preds = self._unpad_predictions(
            predictions=spine_preds,
            original_dimensions=job.original_dimensions,
        )
        dendrite_preds = self._unpad_predictions(
            predictions=dendrite_preds,
            original_dimensions=job.original_dimensions,
        )

        # Overwrite or save output files using the original naming convention.
        self._write_unpadded_predictions(
            job=job,
            spine_preds=spine_preds,
            dendrite_preds=dendrite_preds,
        )

        # Package everything into a structured result object.
        return DeepD3Result(
            backend_name=self.name,
            output_path=job.output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=cmd,
            return_code=completed.returncode,
            runtime_s=runtime_s,
            metadata={
                "model_name": str(job.model_name),
                "device": job.device,
                "images_folder": str(job.input_path),
            },
            spine_probabilities=spine_preds,
            dendrite_probabilities=dendrite_preds,
        )

    def _build_command(self, job: DeepD3Job) -> list[str]:
        """
        Build the DeepD3 subprocess command.

        Parameters
        ----------
        job
            DeepD3 job specification with all necessary parameters.

        Returns
        -------
        list[str]
            Command to execute DeepD3 inference.
        """
        # Verify python executable exists
        python_path = Path(self.python_executable)
        if not python_path.exists():
            raise FileNotFoundError(
                f"Python executable not found: {python_path}\n\n"
                f"Please verify:\n"
                f"1. The DeepD3 conda environment exists:\n"
                f"   conda env list\n"
                f"2. The path in spine_config.yaml 'deepd3_python_executable' "
                f"is correct:\n"
                f"   Check: src/spyne/config/spine_config.yaml\n"
                f"3. If using a different environment name or location, "
                f"update the config file accordingly.\n\n"
                f"Expected path: {python_path}"
            )
        
        cmd = [
            str(self.python_executable),
            "-m", "deepd3.inference.cli",
            "--input", str(job.input_path),
            "--output", str(job.output_dir),
            "--model", str(job.model_name),
            "--device", job.device,
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
        Parse subprocess results into InferenceResult.

        This method satisfies the abstract base class requirement but is not
        used since DeepD3Backend overrides run() entirely. DeepD3 requires
        custom post-processing (unpadding) that doesn't fit the standard
        parse_result pattern.

        Parameters
        ----------
        job
            The inference job that was executed.
        command
            The command that was executed.
        return_code
            The subprocess return code.
        stdout
            Standard output from the subprocess.
        stderr
            Standard error from the subprocess.
        runtime_s
            Runtime in seconds.

        Returns
        -------
        InferenceResult
            Basic result structure (not used in actual workflow).
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

    def _load_predictions(
        self, job: DeepD3Job
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load spine and dendrite probability maps from DeepD3 output.

        DeepD3 writes outputs to 'spines' and 'dendrites' subfolders within
        the output directory, preserving the input filenames.

        Parameters
        ----------
        job
            DeepD3 job specification containing output directory path.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (spine_probabilities, dendrite_probabilities) as 3D arrays.

        Raises
        ------
        FileNotFoundError
            If expected output files are not found.
        """
        spine_dir = job.output_dir / "spines"
        dendrite_dir = job.output_dir / "dendrites"

        if not spine_dir.exists():
            raise FileNotFoundError(f"Spine output not found: {spine_dir}")
        if not dendrite_dir.exists():
            raise FileNotFoundError(
                f"Dendrite output not found: {dendrite_dir}"
            )

        # Get input image files to match output filenames
        # DeepD3 preserves the full input filename including channel suffix
        input_images = sorted(job.input_path.glob('*_0000.tif'))
        n_images = len(input_images)

        if n_images == 0:
            raise FileNotFoundError(
                f"No input images found in {job.input_path}"
            )

        spine_preds = [None] * n_images
        dendrite_preds = [None] * n_images

        # Load predictions matching input filenames
        for idx, img_file in enumerate(input_images):
            # DeepD3 preserves the full input filename
            # Input: dataset_0000_0000.tif -> Output: dataset_0000_0000.tif
            spine_path = spine_dir / f"{img_file.stem}.tif"
            dendrite_path = dendrite_dir / f"{img_file.stem}.tif"

            if not spine_path.exists():
                raise FileNotFoundError(
                    f"Spine prediction file not found: {spine_path}"
                )
            if not dendrite_path.exists():
                raise FileNotFoundError(
                    f"Dendrite prediction file not found: {dendrite_path}"
                )

            spine_preds[idx] = tifffile.imread(spine_path)
            dendrite_preds[idx] = tifffile.imread(dendrite_path)

        return np.array(spine_preds), np.array(dendrite_preds)

    def _unpad_predictions(
        self,
        predictions: np.ndarray,
        original_dimensions: tuple[int, ...] | list[int],
    ) -> np.ndarray:
        """
        Crop padded predictions back to original image dimensions.

        DeepD3 requires input images to be padded to specific sizes. This
        method removes the padding to return predictions matching the
        original input dimensions.

        Uses the existing utilities from segmentation.utils module.

        Parameters
        ----------
        predictions
            Padded prediction array with shape (Z, Y, X) or (Y, X).
        original_dimensions
            Original dimensions to crop to.

        Returns
        -------
        np.ndarray
            Unpadded predictions matching original dimensions.
        """
        return unpad_predictions(predictions, original_dimensions)

    def _write_unpadded_predictions(
        self,
        job: DeepD3Job,
        spine_preds: np.ndarray,
        dendrite_preds: np.ndarray,
    ) -> None:
        """
        Save unpadded predictions to disk, overwriting padded versions.

        Removes the channel suffix (_0000) from filenames to match the
        original naming convention used in the pipeline.

        Parameters
        ----------
        job
            DeepD3 job specification containing output directory.
        spine_preds
            Unpadded spine probability maps.
        dendrite_preds
            Unpadded dendrite probability maps.
        """
        spine_dir = job.output_dir / "spines"
        dendrite_dir = job.output_dir / "dendrites"

        # Get input image files to construct output filenames
        input_images = sorted(job.input_path.glob('*_0000.tif'))
        n_images = len(input_images)

        if n_images != len(spine_preds):
            raise ValueError(
                f"Number of input images ({n_images}) does not match "
                f"number of predictions ({len(spine_preds)})"
            )

        # Clear existing files
        for f in spine_dir.glob("*.tif*"):
            f.unlink()
        for f in dendrite_dir.glob("*.tif*"):
            f.unlink()

        # Save unpadded predictions with channel suffix removed
        for idx, img_file in enumerate(input_images):
            # Remove channel suffix for output filenames
            # Input: dataset_0000_0000.tif -> Output: dataset_0000.tif
            output_stem = img_file.stem[:-5]  # Remove '_0000' suffix
            spine_path = spine_dir / f"{output_stem}.tif"
            dendrite_path = dendrite_dir / f"{output_stem}.tif"

            tifffile.imwrite(
                spine_path,
                spine_preds[idx],
                compression="zlib",
            )
            tifffile.imwrite(
                dendrite_path,
                dendrite_preds[idx],
                compression="zlib",
            )
