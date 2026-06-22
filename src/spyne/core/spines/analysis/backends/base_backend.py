from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import subprocess
import time


# ---------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------

# These data classes are defined here to avoid circular imports and to provide
# a common interface for all backends. Each backend can define its own specific
# job and result classes that inherit from these base classes if needed, or they
# can use them directly if they fit the requirements. This way, we maintain a
# clear separation of concerns while still allowing for flexibility in how each
# backend handles its specific needs.

@dataclass
class InferenceJob:
    """
    Standardized input specification for an inference job across all backends.
    This class captures the essential information needed to run an inference task,
    while allowing for backend-specific extensions through the `extra_args` field.
    
    Attributes:
    ----------
    input_path : Path
        The path to the input data (e.g. image or folder of images).
    output_dir : Path
        The directory where the backend should write its outputs.
    model_name : str | None
        The name of the model to use for inference, if applicable.
    config_path : Path | None
        The path to a configuration file for the backend, if applicable.
    extra_args : dict[str, Any]
        Additional backend-specific arguments.
    """
    input_path: Path
    output_dir: Path
    model_name: str | None = None
    config_path: Path | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def _normalize_job_paths(self) -> InferenceJob:
        """
        Normalize all path fields to absolute paths.

        This ensures that the backend can reliably access the specified
        files and directories regardless of the current working directory.

        Returns
        -------
        InferenceJob
            A new instance of the same job type with paths resolved.
        """
        # Prepare base fields that all InferenceJob types share
        normalized_fields = {
            'input_path': self.input_path.resolve(),
            'output_dir': self.output_dir.resolve(),
            'model_name': self.model_name,
            'config_path': (
                self.config_path.resolve()
                if self.config_path
                else None
            ),
            'extra_args': self.extra_args,
        }
        
        # Preserve subclass-specific fields using __dict__
        for field_name, field_value in self.__dict__.items():
            if field_name not in normalized_fields:
                normalized_fields[field_name] = field_value
        
        # Return new instance of the same type (e.g., DeepD3Job)
        return type(self)(**normalized_fields)


@dataclass
class InferenceResult:
    """
    Standardized output from an inference job across all backends.
    This class captures the essential information about the inference execution,
    including success status, output paths, and execution metadata.
    
    Attributes:
    ----------
    backend_name : str
        The name of the backend that produced this result.
    success : bool
        Whether the inference job was successful.
    output_path : Path | None
        The path to the output produced by the inference job, if any.
    stdout : str
        The standard output captured from the backend process.
    stderr : str
        The standard error captured from the backend process.
    command : list[str]
        The command that was executed for the inference job.
    return_code : int
        The return code from the backend process.
    runtime_s : float
        The runtime of the inference job in seconds.
    metadata : dict[str, Any]
        Additional metadata about the inference job.
    """
    backend_name: str
    output_path: Path | None
    stdout: str
    stderr: str
    command: list[str]
    return_code: int
    runtime_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the inference job succeeded (return code 0)."""
        return self.return_code == 0


# ---------------------------------------------------------------------
# Base backend
# ---------------------------------------------------------------------

# This is the base class for all backends, defining the interface and common
# functionality. Each specific backend (e.g. DeepD3) will inherit from this class
# and implement the required methods. This design allows us to keep the core
# analysis pipeline agnostic to the specific inference implementation, while still
# providing a consistent interface for running inference jobs and handling results.


class BaseBackend(ABC):
    """
    Abstract interface for all inference backends.
    Each backend must implement the `run` method, which takes a standardized
    `InferenceJob` and returns an `InferenceResult`.
    
    This design allows us to easily add new backends in the future (e.g. for
    other spine detection models) while keeping the core analysis pipeline
    agnostic to the specific inference implementation.
    
    Backends are expected to handle all aspects of execution, including:
    - validating input and output paths
    - constructing the appropriate command-line invocation
    - parsing stdout/stderr to determine success and extract relevant outputs
    - measuring execution time
    - returning a normalized result that the rest of the pipeline can work with
    
    By enforcing this interface, we can ensure that the core analysis code can
    interact with any backend in a consistent way, making it easier to maintain
    and extend in the future.
    
    Attributes:
    ----------
    name (str):
        A unique identifier for the backend, used in results metadata.
    python_executable (Path):
        The path to the Python executable to use for running the backend.
    
    Methods:
    -------
    run(job: InferenceJob) -> InferenceResult:
        Execute the inference workflow for the given job and return the result.
    
    """

    name: str

    def __init__(self, python_executable: Path) -> None:
        
        self.python_executable = python_executable

    def run(self, job: InferenceJob) -> InferenceResult:
        
        self._validate_job(job)
        cmd = self._build_command(job)

        start = time.perf_counter()
        completed = self._run_subprocess(cmd)
        runtime_s = time.perf_counter() - start

        result = self._parse_result(
            job=job,
            command=cmd,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_s=runtime_s,
        )
        return result
    
    def _run_subprocess(
        self,
        cmd: list[str],
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """
        Common subprocess execution with error handling.
        This method can be used by subclasses to run their specific commands while
        benefiting from consistent error handling and timeout support.
        """
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Handle errors in subclass
            timeout=timeout,
        )

    def _validate_job(self, job: InferenceJob) -> None:
        """
        Validate that the input and output paths are correct and accessible.
        This method can be overridden by specific backends if they have additional
        requirements (e.g. specific file formats, config files, etc.).
        """
        
        if not job.input_path.exists():
            raise FileNotFoundError(f"Input not found: {job.input_path}")
        
        job.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_output_folder(self, output_dir: Path) -> None:
        """
        Prepare output folder by removing existing content and creating fresh.

        This ensures a clean state for each inference run. Override in subclass
        if different behavior is needed (e.g., incremental outputs).

        Parameters
        ----------
        output_dir
            The directory to prepare for output.
        """
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _build_command(self, job: InferenceJob) -> list[str]:
        """
        Return the subprocess command for this backend.
        Each backend must implement this method to construct the appropriate command-line
        invocation based on the standardized `InferenceJob`. This allows the `run`
        method to remain generic and handle execution and timing, while the specifics
        of how to run the backend are encapsulated in this method.

        Attributes:
        ----------
        job : InferenceJob
            The standardized job specification containing all necessary information for
            constructing the command.
        """

    @abstractmethod
    def _parse_result(
        self,
        job: InferenceJob,
        command: list[str],
        return_code: int,
        stdout: str,
        stderr: str,
        runtime_s: float,
    ) -> InferenceResult:
        """Convert subprocess outputs into a normalized result."""


class BackendError(Exception):
    """Base exception for backend execution failures."""
    def __init__(
        self,
        message: str,
        cmd: list[str],
        stdout: str,
        stderr: str,
        return_code: int,
    ):
        super().__init__(message)
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code