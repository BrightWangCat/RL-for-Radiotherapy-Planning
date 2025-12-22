from typing import Any, Optional, Annotated, Union, Final
from typing_extensions import Self

# import warnings
import numpy as np
from pydantic import (
    Field,
    StringConstraints,
    model_validator,
    field_validator,
    # ValidatorFunctionWrapHandler,
    ValidationInfo,
    ValidationError,
    AliasChoices,
    computed_field,
)
from numpydantic import NDArray, Shape
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.util.helpers import dl2ld
from pyRadPlan.machines._base import ExternalBeamMachine


class IonBeamEmittance(PyRadPlanBaseModel):
    """Emittance data for ion accelerator."""

    type: Final[str] = "bigaussian"

    sigma_x: float = Field(alias="sigmaX", ge=0.0)
    sigma_y: float = Field(alias="sigmaY", ge=0.0)
    div_x: float = Field(default=1e-12, alias="divX")
    div_y: float = Field(default=1e-12, alias="divY")
    corr_x: float = Field(default=1e-12, alias="corrX", ge=-1.0, le=1.0)
    corr_y: float = Field(default=1e-12, alias="corrY", ge=-1.0, le=1.0)


class IonEnergySpectrum(PyRadPlanBaseModel):
    pass


class IonEnergySpectrumGaussian(IonEnergySpectrum):
    """
    Energy spectrum data for ion pencil beam kernels.

    Notes
    -----
    The energy spectrum is defined as a Gaussian distribution with a mean
    energy and a relative spread in percent.
    """

    type: Final[str] = "gaussian"

    mean: float = Field(ge=0.0)
    sigma: float = Field(
        ge=0.0, description="Relative spread as percentage sigma of the Gaussian energy spectrum"
    )

    @property
    def sigma_relative(self) -> float:
        """Relative spread of the energy spectrum."""
        return self.sigma / 100.0

    @property
    def sigma_absolute(self) -> float:
        """Absolute sigma of the energy spectrum."""
        return self.sigma_relative * self.mean

    @property
    def fwhm(self) -> float:
        """Full width at half maximum of the energy spectrum."""
        return 2.35482 * self.sigma_absolute


class IonBeamFocus(PyRadPlanBaseModel):
    """Focus data for ion pencil beam kernels."""

    dist: NDArray[Shape["1-*"], np.float64]
    sigma: NDArray[Shape["1-*"], np.float64]
    fwhm_iso: Optional[np.float64] = None

    # emittance parameterization
    emittance: Optional[IonBeamEmittance] = None

    @field_validator("dist", "sigma", mode="before")
    @classmethod
    def validate_arrays(cls, v: Any) -> Any:
        """Validate the focus distance and sigma arrays."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

    @property
    def has_emittance(self) -> bool:
        """Check if emittance parameters are available."""
        return self.sigma_x is not None and self.sigma_y is not None


class LateralCutOff(PyRadPlanBaseModel):
    """Lateral cut-off radius and compensation for ion pencil beam kernels."""

    comp_fac: np.float64 = Field(default=np.float64(1.0))
    cut_off: NDArray[Shape["1-*"], np.float64] = Field(
        default=np.array([np.inf, np.inf], dtype=np.float64)
    )
    depths: NDArray[Shape["1-*"], np.float64] = Field(
        default=np.array([0.0, np.inf], dtype=np.float64)
    )

    @field_validator("comp_fac", mode="before")
    @classmethod
    def validate_comp_fac(cls, v: Any) -> Any:
        """Validate the lateral cut-off compensation factor."""
        try:
            return np.float64(v)
        except ValueError as exc:
            raise exc

    @field_validator("cut_off", mode="before")
    @classmethod
    def validate_arrays(cls, v: Any) -> Any:
        """Validate the lateral cut-off radial distance and depths."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v


class IonPencilBeamKernel(PyRadPlanBaseModel):
    """Data Model for a single Ion Pencil Beam Kernel."""

    energy: np.float64
    peak_pos: np.float64
    range: Optional[np.float64] = None
    offset: np.float64 = Field(default=np.float64(0.0))
    depths: NDArray[Shape["1-*"], np.float64]

    # all kernels
    idd: NDArray[Shape["1-*"], np.float64] = Field(
        validation_alias=AliasChoices("Z", "z"), serialization_alias="Z"
    )
    sigma: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_1: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_2: Optional[NDArray[Shape["1-*"], np.float64]] = None
    weight: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_multi: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    weight_multi: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    let: Optional[NDArray[Shape["1-*"], np.float64]] = Field(alias="LET", default=None)
    alpha_x: Optional[NDArray[Shape["1-*"], np.float64]] = None
    beta_x: Optional[NDArray[Shape["1-*"], np.float64]] = None
    alpha: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    beta: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    lateral_cut_off: Optional[LateralCutOff] = None

    @field_validator("energy", "peak_pos", "range", "offset", mode="before")
    @classmethod
    def validate_possible_cast(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64."""
        try:
            return np.float64(v)
        except ValueError:
            return v

    @field_validator(
        "depths",
        "idd",
        "sigma",
        "sigma_1",
        "sigma_2",
        "weight",
        "sigma_multi",
        "weight_multi",
        "let",
        "alpha",
        "beta",
        "alpha_x",
        "beta_x",
        mode="before",
    )
    @classmethod
    def validate_possible_cast_array(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64 array."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

    @field_validator("idd", "sigma", "sigma_1", "sigma_2", "let", mode="after")
    @classmethod
    def validate_kernel_lengths(
        cls, v: Union[np.ndarray, None], info: ValidationInfo
    ) -> Union[np.ndarray, None]:
        """Validate the length of the kernel data."""
        if v is None:
            return v

        if v.shape != info.data["depths"].shape:
            raise ValueError("Kernel data length does not match the depth data length.")

        return v

    @field_validator("sigma_multi", "weight_multi", mode="after")
    @classmethod
    def validate_multi_gaussian_kernel(
        cls, v: Union[np.ndarray, None], info: ValidationInfo
    ) -> Union[np.ndarray, None]:
        """Validate the length of the multi-gaussian kernel data."""
        if v is None:
            return v

        if v.shape[0] != info.data["depths"].shape[0]:
            raise ValueError("Kernel data length does not match the depth data length.")

        return v

    @field_validator("alpha", "beta", mode="after")
    @classmethod
    def validate_alpha_beta(
        cls, v: Union[NDArray[Shape["1-*"], np.float64], None], info: ValidationInfo
    ) -> Union[NDArray[Shape["1-*"], np.float64], None]:
        """Validate the length of the alpha-beta kernel data."""
        if v is None:
            return v

        # Check if kernel is transposed but oterhwise of correct shape
        if (
            v.shape[0] == info.data["depths"].shape[0]
            and v.shape[1] == info.data["alpha_x"].size
            and v.shape[1] == info.data["beta_x"].size
        ):
            v = np.ascontiguousarray(v.T)

        if v.shape[0] != info.data["alpha_x"].size and v.shape[0] != info.data["beta_x"].size:
            raise ValueError(
                "Kernel data length does not match the number of reference photon alpha / beta "
                "values."
            )

        if v.shape[1] != info.data["depths"].size:
            raise ValueError("Kernel data length does not match the depth data length.")

        return v

    @computed_field(return_type=NDArray[Shape["1-*"], np.float64], alias="alphaBetaRatio")
    @property
    def alpha_beta_ratio(self) -> Union[NDArray[Shape["1-*"], np.float64], None]:
        """Photon alpha-beta ratio for each alpha / beta kernel."""
        if self.alpha is None or self.beta is None:
            return None
        return self.alpha_x / self.beta_x


class IonAccelerator(ExternalBeamMachine):
    """
    Machine Model for Ion Accelerators.

    Defines minimum meta-data an ion machine must hold
    Provides multiple data storage formats, currently supported
    - Pencil-Beam Kernels
    - Monte Carlo Beam Models.

    Attributes
    ----------
    sad : float
        The source-to-axis (-isocenter) distance of the machine
    """

    radiation_mode: Annotated[str, StringConstraints(pattern="^(protons|helium|carbon)$")] = Field(
        default="protons", validate_default=True
    )

    sad: float = Field(ge=0.0, description="Source-to-axis distance", alias="SAD")
    bams_to_iso_dist: float = Field(
        ge=0.0,
        description="Beam-monioring-system/nozzle to iso center distance",
        alias="BAMStoIsoDist",
    )
    lut_spot_size: NDArray[Shape["2,*"], np.float64] = Field(
        description="2D array of look up table latteral spotspacing to foxus index",
        alias=AliasChoices(
            "LUTspotSize", "LUT_bxWidthminFWHM"
        ),  # We have two aliases here to capture version changes in matRad base data files
    )
    # kernel_data: Annotated[np.recarray, Field(description="Pencil beam kernel data")]

    fit_air_offset: float = Field(
        ge=0.0,
        default=0.0,
        description="Air distance included in fitting of the kernel",
    )

    # Minimum required data is energies, Bragg-peak positions and beam foci
    peak_positions: NDArray[Shape["1-*"], np.float64]
    foci: dict[float, list[IonBeamFocus]]
    spectra: Optional[dict[float, IonEnergySpectrum]] = None

    # Optionally we can have pencil-beam kernels
    pb_kernels: Optional[dict[float, IonPencilBeamKernel]] = None

    @field_validator("lut_spot_size", mode="before")
    @classmethod
    def validate_possible_cast_array(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64 array."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

    # TODO: Beam model descriptions for Monte Carlo

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""

        super(IonAccelerator, cls)._parse_tabulated_energy_data_from_mat(
            tabulated_energy_data, returned_data
        )

        # extrac required quantities
        if "offset" in tabulated_energy_data:
            returned_data["peak_positions"] = np.array(
                tabulated_energy_data["peakPos"], dtype=np.float64
            ) + np.array(tabulated_energy_data["offset"], dtype=np.float64)

        # extract beam foci
        foci = {}
        # First, check if we have the special situation of only a single focus

        foci_list = tabulated_energy_data["initFocus"]
        if not isinstance(foci_list, list):
            foci_list = [foci_list]

        for i, foci_entry in enumerate(foci_list):
            # check if all initFocus fields are lists
            if not all(isinstance(foci_entry[key], list) for key in foci_entry):
                focus_list = [foci_entry]
            else:
                focus_list = dl2ld(foci_entry)
            foci[returned_data["energies"][i]] = [IonBeamFocus(**focus) for focus in focus_list]

        returned_data["foci"] = foci

        # Energy Spectra
        spectra_list = tabulated_energy_data.get("energySpectrum", None)
        if spectra_list is not None:
            returned_data["spectra"] = {
                returned_data["energies"][i]: IonEnergySpectrumGaussian(**spectrum)
                for i, spectrum in enumerate(spectra_list)
            }

        # Extract the pencil beam kernels
        # Pencil beam kernels are not requried so we may skip them
        # if they can't be validated
        try:
            tmp = dl2ld(tabulated_energy_data)
            returned_data["pb_kernels"] = {
                entry["energy"]: IonPencilBeamKernel(**entry) for entry in tmp
            }
        except ValidationError:
            returned_data["pb_kernels"] = None

    # @model_validator(mode="after")
    # def validate_machine_components(self):
    #    pass
    # if self.betas not None:

    @model_validator(mode="after")
    def _check_machine(self) -> Self:
        """Validate the machine model for consistency."""

        if self.bams_to_iso_dist > self.sad:
            raise ValueError("BAMS to iso distance must be small than SAD.")

        if self.pb_kernels is not None:
            # Check that the peak_positions are consistent between peak_positions of the model
            # and the kernel peak positions with offset applied
            for i, energy in enumerate(self.energies):
                kernel = self.pb_kernels[energy]
                if not np.isclose(self.peak_positions[i], kernel.peak_pos + kernel.offset):
                    raise ValueError(
                        f"Peak position of the model and the kernel data for energy {energy} "
                        "are inconsistent."
                    )

        return self

    @property
    def has_pb_kernels(self):
        """Test for existance of valid pencil-beam kernels."""
        return self.pb_kernels is not None

    @property
    def pb_kernels_have_let(self) -> bool:
        """Check if the kernel data contains LET values."""
        return all(kernel.let is not None for kernel in self.pb_kernels.values())

    def get_kernel_by_index(self, ix_energy: int) -> IonPencilBeamKernel:
        """Get the pencil beam kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        IonPencilBeamKernel
            Pencil beam kernel data
        """
        return self.get_kernel_by_energy(self.energies[ix_energy])

    def get_foci_by_index(self, ix_energy: int) -> list[IonBeamFocus]:
        """
        Get the focus data of the kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        IonBeamFocus
            Dictionary of available beam foci
        """
        return self.foci[self.energies[ix_energy]]

    def get_kernel_by_energy(self, energy: float) -> IonPencilBeamKernel:
        """Get the pencil beam kernel for a specific energy value.

        Parameters
        ----------
        energy : float
            Energy value to search for

        Returns
        -------
        IonPencilBeamKernel
            Pencil beam kernel data
        """
        if self.pb_kernels is None:
            raise ValueError("No pencil beam kernels available for this machine.")
            # return None
        return self.pb_kernels[energy]

    def update_kernel_at_index(self, ix_energy: int, kernel: IonPencilBeamKernel):
        """Update the pencil beam kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data
        kernel : IonPencilBeamKernel
            Pencil beam kernel data
        """

        self.update_kernel_for_energy(self.energies[ix_energy], kernel)

    def update_kernel_for_energy(
        self, energy: Union[float, np.float64], kernel: IonPencilBeamKernel
    ):
        """
        Update a pencil-beam kernel for a specific energy.

        Parameters
        ----------
        energy : float
            Energy value to update the kernel for
        kernel : IonPencilBeamKernel
            The Kernel to replace the existing one

        Notes
        -----
        If your intention is to build a new machine / add kernels to a machine without exisiting
        kernels, you should create the machine from scratch or set the whole model field at once
        which is a dictionary of all kernels with the energy as key
        """
        if self.pb_kernels is None:
            raise ValueError("No pencil beam kernels available for this machine.")
        energy = np.float64(energy)

        self.pb_kernels[energy] = kernel

    def get_focus(self, ix_energy: int) -> dict:
        """
        Get the focus data of the kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        dict
            Dictionary of focus data
        """

        return self.foci[self.energies[ix_energy]]

    @property
    def has_single_gaussian_kernel(self) -> bool:
        """
        Check if it has single gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have single gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(kernel.sigma is not None for kernel in self.pb_kernels.values())

    @property
    def has_double_gaussian_kernel(self) -> bool:
        """
        Check if it has double gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have double gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(
            kernel.sigma_1 is not None and kernel.sigma_2 is not None and kernel.weight is not None
            for kernel in self.pb_kernels.values()
        )

    @property
    def has_multi_gaussian_kernel(self) -> bool:
        """
        Check if it has multi gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have multi gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(
            kernel.sigma_multi is not None and kernel.weight_multi is not None
            for kernel in self.pb_kernels.values()
        )

    @property
    def has_let_kernel(self) -> bool:
        """
        Check if it has LET values in the pencil-beam kernels.

        Returns
        -------
        bool
            True if all kernels have LET values
        """
        if self.pb_kernels is None:
            return False

        return all(kernel.let is not None for kernel in self.pb_kernels.values())

    @property
    def has_alpha_beta_kernels(self) -> bool:
        """
        Check if it has alpha-beta kernels.

        Returns
        -------
        bool
            True if all kernels have alpha-beta values
        """

        if self.pb_kernels is None:
            return False

        return all(
            kernel.alpha is not None and kernel.beta is not None
            for kernel in self.pb_kernels.values()
        )
