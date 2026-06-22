from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil
import json
import os
import time
import subprocess

import numpy as np
import tifffile

from spyne.core.spines.analysis.backends.base_backend import (
    BaseBackend, InferenceJob, InferenceResult)
from spyne.core.spines.analysis.padding import unpad_predictions


# ---------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------

@dataclass
class NNUNetJob(InferenceJob):
    """
    Input specification for one nnU-Net (RESPAN) batch inference run.

    nnU-Net has specific requirements:
    - input is a folder of padded TIFF images with _0000 suffix
    - requires nnUNet directory structure or creates temporary one
    - outputs both probability maps (.npz) and label maps (.tif)
    - returns binary masks via argmax (no thresholding needed)
    """
    original_dimensions: tuple[int, ...] | list[int] = field(
        default_factory=list
    )
    dataset_id: str = "100"                 # Just a placeholder ID
    dataset_name: str = "SpineSegmentation" # Placeholder name
    configuration: str = "2d"


@dataclass
class NNUNetResult(InferenceResult):
    """
    Normalized result returned by the nnU-Net backend.

    Extends InferenceResult with nnU-Net-specific outputs:
    - spine_probabilities: probability maps for detected spines
    - dendrite_probabilities: probability maps for detected dendrites
    - spine_binary_masks: binary masks from argmax (0/1)
    - dendrite_binary_masks: binary masks from argmax (0/1)

    All other fields inherited from InferenceResult.
    """
    spine_probabilities: np.ndarray | None = None
    dendrite_probabilities: np.ndarray | None = None
    spine_binary_masks: np.ndarray | None = None
    dendrite_binary_masks: np.ndarray | None = None


# ---------------------------------------------------------------------
# nnU-Net backend
# ---------------------------------------------------------------------

class NNUNetBackend(BaseBackend):
    """
    Backend wrapper for nnU-Net (RESPAN) batch inference.

    This class encapsulates:
    - nnUNet directory structure creation (if needed)
    - command construction for nnUNetv2_predict
    - subprocess execution with environment variables
    - loading probability maps from .npz files
    - extracting binary masks from multi-class predictions
    - unpadding predictions
    - organizing outputs to match DeepD3 structure
    """

    name = "nnunet"

    def __init__(self, python_executable: Path | str) -> None:
        """
        Parameters
        ----------
        python_executable
            Python interpreter inside the environment where nnUNet is
            installed.
        """
        self.python_executable = Path(python_executable).resolve()
        self._temp_dir = None  # Track temporary directory for cleanup

    def run(self, job: NNUNetJob) -> NNUNetResult:
        """
        Run the full nnU-Net inference workflow.

        This method is the public entry point used by the rest of the
        pipeline.
        """
        # Normalize and validate paths
        job = job._normalize_job_paths()
        self._validate_job(job)

        # Prepare output directories
        self._prepare_output_folder(job.output_dir)
        
        # Create temporary probability output folder for nnUNet raw outputs
        temp_prediction_dir = job.output_dir.parent / "temp_nnunet_predictions"
        self._prepare_output_folder(temp_prediction_dir)

        # Setup nnUNet directory structure
        nnunet_results_dir = self._setup_nnunet_structure(job)

        # Build command
        cmd = self._build_command(job, temp_prediction_dir)

        # Execute nnUNet with environment variables
        start_time = time.perf_counter()
        completed = self._run_subprocess_with_env(
            cmd, nnunet_results_dir
        )
        runtime_s = time.perf_counter() - start_time

        # Check if nnUNet succeeded before trying to load predictions
        if completed.returncode != 0:
            raise RuntimeError(
                f"nnUNet inference failed with return code "
                f"{completed.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        # Load and process predictions
        (spine_probs, dendrite_probs,
         spine_masks, dendrite_masks) = self._load_and_process_predictions(
            job, temp_prediction_dir
        )

        # Organize probability maps into spines/dendrites structure
        self._organize_probability_maps(
            job, spine_probs, dendrite_probs
        )

        # Clean up temporary directories
        shutil.rmtree(temp_prediction_dir, ignore_errors=True)
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        # Package results
        return NNUNetResult(
            backend_name=self.name,
            output_path=job.output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=cmd,
            return_code=completed.returncode,
            runtime_s=runtime_s,
            metadata={
                "model_name": str(job.model_name),
                "dataset_id": job.dataset_id,
                "dataset_name": job.dataset_name,
                "configuration": job.configuration,
            },
            spine_probabilities=spine_probs,
            dendrite_probabilities=dendrite_probs,
            spine_binary_masks=spine_masks,
            dendrite_binary_masks=dendrite_masks,
        )

    def _build_command(
        self, job: NNUNetJob, output_dir: Path
    ) -> list[str]:
        """
        Build the nnUNetv2_predict command.

        Parameters
        ----------
        job
            nnU-Net job specification.
        output_dir
            Directory for nnUNet to write predictions.

        Returns
        -------
        list[str]
            Command to execute nnUNetv2_predict.
        """
        # Find nnUNetv2_predict executable
        python_path = Path(self.python_executable)
        
        # Verify python executable exists
        if not python_path.exists():
            raise FileNotFoundError(
                f"Python executable not found: {python_path}\n\n"
                f"Please verify:\n"
                f"1. The nnU-Net conda environment exists:\n"
                f"   conda env list\n"
                f"2. The path in spine_config.yaml 'nnunet_python_executable' "
                f"is correct:\n"
                f"   Check: src/spyne/config/spine_config.yaml\n"
                f"3. If using a different environment name or location, "
                f"update the config file accordingly.\n\n"
                f"Expected path: {python_path}"
            )
        
        # Try multiple possible locations and names
        possible_executables = [
            python_path.parent / "nnUNetv2_predict.exe",
            python_path.parent / "nnUNetv2_predict",
            python_path.parent / "Scripts" / "nnUNetv2_predict.exe",
            python_path.parent / "Scripts" / "nnUNetv2_predict",
        ]
        
        nnunet_predict = None
        for exe in possible_executables:
            if exe.exists():
                nnunet_predict = exe
                break
        
        if nnunet_predict is None:
            # If executable not found, try running as Python module
            print(f"nnUNetv2_predict executable not found, using Python module approach")
            print(f"Python executable: {python_path}")
            cmd = [
                str(python_path),
                "-m", "nnunetv2.inference.predict",
                "-i", str(job.input_path),
                "-o", str(output_dir),
                "-d", job.dataset_id,
                "-c", job.configuration,
                "-f", "all",
                "--save_probabilities"
            ]
        else:
            print(f"Using nnUNetv2_predict executable: {nnunet_predict}")
            cmd = [
                str(nnunet_predict),
                "-i", str(job.input_path),
                "-o", str(output_dir),
                "-d", job.dataset_id,
                "-c", job.configuration,
                "-f", "all",
                "--save_probabilities"
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

        nnU-Net requires custom post-processing that doesn't fit the
        standard pattern.
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

    def _setup_nnunet_structure(self, job: NNUNetJob) -> Path:
        """
        Setup nnUNet directory structure.

        If model_name points to a checkpoint file, creates temporary
        nnUNet-compatible directory structure. If it points to a trainer
        directory, validates and uses existing structure.

        Parameters
        ----------
        job
            nnU-Net job specification.

        Returns
        -------
        Path
            Path to nnUNet_results directory.
        """
        model_path = Path(job.model_name)

        if model_path.is_file():
            # Create temporary directory structure
            self._temp_dir = Path(
                tempfile.mkdtemp(prefix="nnunet_inference_")
            )

            nnunet_results_dir = self._temp_dir / "nnUNet_results"
            dataset_dir = (
                nnunet_results_dir
                / f"Dataset{job.dataset_id}_{job.dataset_name}"
            )
            trainer_dir = (
                dataset_dir
                / f"nnUNetTrainer__nnUNetPlans__{job.configuration}"
            )
            fold_dir = trainer_dir / "fold_all"

            fold_dir.mkdir(parents=True, exist_ok=True)

            # Find and copy plans file
            plans_json = model_path.parent / "plans.json"
            if not plans_json.exists():
                plans_json = model_path.parent / "nnUNetPlans.json"
                if not plans_json.exists():
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                    raise FileNotFoundError(
                        f"plans.json or nnUNetPlans.json not found in "
                        f"{model_path.parent}\n"
                        f"nnUNet requires both checkpoint and plans file."
                    )

            # Copy checkpoint and plans
            shutil.copy2(model_path, fold_dir / model_path.name)
            shutil.copy2(plans_json, trainer_dir / "plans.json")

            # Create minimal dataset.json
            dataset_json = {
                "channel_names": {"0": "calcium"},
                "labels": {
                    "background": 0,
                    "spines": 1,
                    "dendrites": 2
                },
                "numTraining": 0,
                "file_ending": ".tif"
            }
            with open(trainer_dir / "dataset.json", 'w') as f:
                json.dump(dataset_json, f, indent=2)

            return nnunet_results_dir

        elif model_path.is_dir():
            # Validate existing trainer directory
            trainer_dir = model_path

            plans_json = trainer_dir / "plans.json"
            if not plans_json.exists():
                plans_json = trainer_dir / "nnUNetPlans.json"
                if not plans_json.exists():
                    raise FileNotFoundError(
                        f"plans.json not found in {trainer_dir}"
                    )

            # Find nnUNet_results (2 levels up)
            dataset_dir = trainer_dir.parent
            nnunet_results_dir = dataset_dir.parent

            if nnunet_results_dir.name != "nnUNet_results":
                nnunet_results_dir = trainer_dir

            return nnunet_results_dir

        else:
            raise ValueError(
                f"Model path must be file or directory: {model_path}"
            )

    def _run_subprocess_with_env(
        self,
        cmd: list[str],
        nnunet_results_dir: Path
    ) -> subprocess.CompletedProcess:
        """
        Run subprocess with nnUNet environment variables.

        Parameters
        ----------
        cmd
            Command to execute.
        nnunet_results_dir
            Path to set as nnUNet_results environment variable.

        Returns
        -------
        subprocess.CompletedProcess
            Completed process result.
        """
        import subprocess
        
        env = os.environ.copy()
        env['nnUNet_results'] = str(nnunet_results_dir)

        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Failed to execute command: {' '.join(cmd)}\n"
                f"First element (executable): {cmd[0]}\n"
                f"Error: {e}"
            ) from e

    def _load_and_process_predictions(
        self,
        job: NNUNetJob,
        prediction_dir: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load probability maps and binary masks from nnUNet output.

        Parameters
        ----------
        job
            nnU-Net job specification.
        prediction_dir
            Directory containing nnUNet outputs.

        Returns
        -------
        tuple
            (spine_probs, dendrite_probs, spine_masks, dendrite_masks)
        """
        input_images = sorted(job.input_path.glob('*_0000.tif'))
        n_images = len(input_images)

        if n_images == 0:
            raise FileNotFoundError(
                f"No input images found in {job.input_path}"
            )

        spine_probs = [None] * n_images
        dendrite_probs = [None] * n_images
        spine_masks = [None] * n_images
        dendrite_masks = [None] * n_images

        for idx, img_file in enumerate(input_images):
            # nnUNet removes channel suffix from output
            output_stem = img_file.stem[:-5]

            # Load probability file
            prob_file = prediction_dir / f"{output_stem}.npz"
            if prob_file.exists():
                with np.load(prob_file) as prob_data:
                    prob_key = (
                        'probabilities' if 'probabilities' in prob_data
                        else 'softmax'
                    )
                    probs = prob_data[prob_key]
                    probs = np.squeeze(probs)

                    # Extract class probabilities (still padded)
                    spine_probs[idx] = probs[1].copy()
                    dendrite_probs[idx] = probs[2].copy()

                prob_file.unlink()

            # Load label map
            pred_file = prediction_dir / f"{output_stem}.tif"
            if not pred_file.exists():
                raise FileNotFoundError(
                    f"Prediction file not found: {pred_file}"
                )

            prediction = tifffile.imread(pred_file)

            # Extract binary masks from multi-class label map
            spine_masks[idx] = (prediction == 1).astype(float)
            dendrite_masks[idx] = (prediction == 2).astype(float)

        # Stack and unpad
        spine_probs = unpad_predictions(
            np.array(spine_probs), job.original_dimensions
        )
        dendrite_probs = unpad_predictions(
            np.array(dendrite_probs), job.original_dimensions
        )
        spine_masks = unpad_predictions(
            np.array(spine_masks), job.original_dimensions
        )
        dendrite_masks = unpad_predictions(
            np.array(dendrite_masks), job.original_dimensions
        )

        return spine_probs, dendrite_probs, spine_masks, dendrite_masks

    def _organize_probability_maps(
        self,
        job: NNUNetJob,
        spine_probs: np.ndarray,
        dendrite_probs: np.ndarray,
    ) -> None:
        """
        Save probability maps in spines/dendrites folder structure.

        Matches DeepD3 output organization for consistency.

        Parameters
        ----------
        job
            nnU-Net job specification.
        spine_probs
            Unpadded spine probability maps.
        dendrite_probs
            Unpadded dendrite probability maps.
        """
        spine_dir = job.output_dir / "spines"
        dendrite_dir = job.output_dir / "dendrites"

        spine_dir.mkdir(parents=True, exist_ok=True)
        dendrite_dir.mkdir(parents=True, exist_ok=True)

        input_images = sorted(job.input_path.glob('*_0000.tif'))

        for idx, img_file in enumerate(input_images):
            # Remove channel suffix to match convention
            output_stem = img_file.stem[:-5]

            tifffile.imwrite(
                spine_dir / f"{output_stem}.tif",
                spine_probs[idx],
                compression="zlib",
            )
            tifffile.imwrite(
                dendrite_dir / f"{output_stem}.tif",
                dendrite_probs[idx],
                compression="zlib",
            )
