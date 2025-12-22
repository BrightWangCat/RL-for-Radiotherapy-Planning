"""Structure Set Implementation."""

from typing import Any, Union
from typing_extensions import Self
from pydantic import Field, model_validator

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.cst import VOI, ExternalVOI, validate_voi
from pyRadPlan.ct import CT, validate_ct


class StructureSet(PyRadPlanBaseModel):
    """Represents a Structure Set for a Patient."""

    vois: list[VOI] = Field(init=False, description="List of VOIs in the Structure Set")
    ct_image: CT = Field(init=False, description="Reference to the CT Image")

    @model_validator(mode="after")
    def check_cst(self) -> Self:
        """Check if the VOIs are valid and reference the same CT."""

        if isinstance(self.vois, list):
            for voi in self.vois:
                if voi.ct_image != self.ct_image:
                    raise ValueError("All VOIs must reference the same CT image.")

        return self

    def to_matrad(self, context: str = "mat-file") -> Any:
        """Convert the StructureSet to a matRad writeable format."""

        if context != "mat-file":
            raise ValueError(f"Context {context} not supported")

        export_cell_list = []
        for i, voi in enumerate(self.vois):
            voi_list = voi.to_matrad(context=context)
            voi_list[0] = i
            # TODO: set objectives here
            voi_list[5] = {}

            export_cell_list.append(voi_list)

        return export_cell_list

    # Additional Properties
    @property
    def voi_types(self) -> list:
        """Return the unique VOI types in the Structure Set."""
        return list({voi.voi_type for voi in self.vois})

    def target_union_voxels(self, order="sitk") -> np.ndarray:
        """Return the union of all target indices."""
        target_indices = []
        for voi in self.vois:
            if voi.voi_type == "TARGET":
                target_indices.append(voi.get_indices(order=order))

        return np.unique(np.concatenate(target_indices))

    def target_union_mask(self) -> sitk.Image:
        """Return the union mask of all target indices."""

        target_indices = self.target_union_voxels(order="numpy")

        # Creates a copy of the CT image with all zeros

        if self.ct_image.cube_hu.GetDimension() == 4:
            sz = np.array(self.ct_image.cube_hu.GetSize())
            sz[3] = 0
            tmpmask3d = sitk.Extract(
                self.ct_image.cube_hu,
                size=sz.tolist(),
                index=[0, 0, 0, 0],
                directionCollapseToStrategy=sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOSUBMATRIX,
            )

        else:
            tmpmask3d = self.ct_image.cube_hu

        mask = sitk.GetArrayViewFromImage(tmpmask3d).astype(np.uint8)
        mask.fill(0)
        mask.ravel()[target_indices] = 1

        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(tmpmask3d)

        return mask_image

    def patient_voxels(self, order="sitk") -> np.ndarray:
        """Return the union of all patient indices."""

        # First check if we have an "EXTERNAL" VOI designating the outer contour
        for voi in self.vois:
            if isinstance(voi, ExternalVOI):
                return voi.get_indices(order=order)

        patient_indices = []
        for voi in self.vois:
            patient_indices.append(voi.get_indices(order=order))

        return np.unique(np.concatenate(patient_indices))

    def patient_mask(self) -> sitk.Image:
        """Return the union mask of all patient contours (or the EXTERNAL
        contour if provided).
        """

        patient_indices = self.patient_voxels(order="numpy")

        # Creates a copy of the CT image with all zeros
        mask = sitk.GetArrayFromImage(self.ct_image.cube_hu).astype(np.uint8)
        mask.fill(0)
        mask.ravel()[patient_indices] = 1

        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(self.ct_image.cube_hu)

        return mask_image

    def target_center_of_mass(self) -> np.ndarray:
        """Return the center of mass of the target."""
        mask_image = self.target_union_mask()
        mask = sitk.GetArrayViewFromImage(
            mask_image
        ).T  # Transpose allows use to use sitk indexing

        if mask.ndim == 4:
            mask = mask[:, :, :, 0]

        cm_index = ndimage.center_of_mass(mask)

        cm = mask_image.TransformContinuousIndexToPhysicalPoint(cm_index)

        return np.array(cm)

    def resample_on_new_ct(self, new_ct: CT) -> Self:
        """
        Resample the StructureSet on a new CT.

        Parameters
        ----------
        new_ct : CT
            The new CT to resample the StructureSet on.

        Returns
        -------
        StructureSet
            The resampled StructureSet.
        """

        new_model = self.model_dump()

        if new_model["ct_image"] != new_ct:
            new_model["ct_image"] = new_ct
            new_model["vois"] = [voi.resample_on_new_ct(new_ct) for voi in self.vois]
        return self.model_validate(new_model)

    def apply_overlap_priorities(self) -> Self:
        """
        Apply overlaps to the StructureSet.

        Returns
        -------
        StructureSet
            The StructureSet with overlaps applied.
        """

        # gather overlaps
        overlaps = [v.overlap_priority for v in self.vois]

        # sort by overlap priority
        ix_sorted = np.argsort(overlaps)

        overlap_mask = self.vois[ix_sorted[0]].mask
        new_vois = [None] * len(self.vois)
        new_vois[ix_sorted[0]] = self.vois[ix_sorted[0]].copy()

        for i, ix_voi in enumerate(ix_sorted[1:], 1):
            curr_voi = self.vois[ix_voi].copy()
            curr_mask = curr_voi.mask

            # if the overlap prirority is higher than we need to apply overlap
            if curr_voi.overlap_priority > new_vois[ix_sorted[i - 1]].overlap_priority:
                curr_mask = sitk.MaskNegated(curr_mask, overlap_mask)

            curr_voi.mask = curr_mask

            overlap_mask = sitk.Or(overlap_mask, curr_mask)

            # sitk.Show(overlap_mask, debugOn = True)

            new_vois[ix_voi] = curr_voi

        return self.model_copy(deep=True, update={"vois": new_vois})


def create_cst(
    cst_data: Union[dict[str, Any], StructureSet, None] = None,
    ct: Union[CT, dict, None] = None,
    **kwargs,
) -> StructureSet:
    """
    Create a StructureSet from various input types.

    Parameters
    ----------
    cst_data : Union[dict[str, Any], StructureSet, None] , optional
        The input data to create the CT object from. Can be a dictionary,
        existing CT object, file path, or None.

    ct : Union[CT,dict,None], CT, None], optional
        The input data to create the CT object from. Can be a dictionary,
        existing CT object, or None.

    **kwargs
        Additional keyword arguments to create the CT object.

    Returns
    -------
    cst
        A StructureSet object created from the input data or keyword arguments.
    """

    # Check if already a valid model
    if isinstance(cst_data, StructureSet):
        if ct is not None and cst_data.ct_image != ct:
            raise ValueError("CT image mismatch between StructureSet and provided CT")
        return cst_data

    # validate ct if present
    if ct is not None:
        ct = validate_ct(ct)

    # If already a model dictionary check the ct setup
    if isinstance(cst_data, dict):
        ct_image = cst_data.pop("ct_image", cst_data.pop("ctImage", None))
        if ct_image is not None:
            cst_data["ct_image"] = ct_image
            if ct is not None and ct != ct_image:
                raise ValueError("CT image mismatch between StructureSet and provided CT")
        elif ct is not None:
            cst_data["ct_image"] = ct
        else:
            raise ValueError("No CT image reference provided!")
        return StructureSet.model_validate(cst_data)

    if cst_data is None and ct is not None:  # If data is None
        return StructureSet(ct_image=ct, **kwargs)
    if cst_data is None and ct is None:
        return StructureSet(**kwargs)

    # Other methods need the CT
    if ct is None:
        raise ValueError("No CT image reference provided!")

    # Creation from an nd array (cell array matRad format)
    if isinstance(cst_data, np.ndarray) and cst_data.dtype == object:
        cst_data = cst_data.to_list()

    # a list of volume information (e.g. imported from pymatreader from matRad mat file)
    if isinstance(cst_data, list):
        voi_list = []
        for vdata in cst_data:
            # First try to read the index lists
            idx_list = []
            # Only one scenario (3D CT)
            if not isinstance(vdata[3], list):
                idx_list.append(np.asarray(vdata[3]).astype(int).tolist())

            # Multiple scenarios (4D CT)
            else:
                for vdata_scen in vdata[3]:
                    idx_list.append(vdata_scen.astype(int).tolist())

            # Now we create isimple ITK masks
            masks = []
            for idx in idx_list:
                # TODO: Check index ordering
                tmp_mask = np.zeros((ct.size[2], ct.size[0], ct.size[1]), dtype=np.uint8)
                tmp_mask.flat[np.asarray(idx) - 1] = 1
                tmp_mask = np.swapaxes(tmp_mask, 1, 2)
                mask_image = sitk.GetImageFromArray(tmp_mask)
                mask_image.CopyInformation(ct.cube_hu)

                masks.append(mask_image)

            # For 4D, we need to join the masks. We also check here if the number of masks we could
            # extract matches the number of dimensions in the CT image
            if ct.cube_hu.GetDimension() == 4:
                # First check if the mask is the same for all 4D scenarios
                if len(masks) == 1:
                    masks = [masks[0] for _ in range(ct.cube_hu.GetDimension())]

                # Now do a sanity check that we don't have an incompatible number of masks
                if len(masks) != ct.cube_hu.GetSize()[3]:
                    raise ValueError("Incompatible number of masks for 4D CT")

                masks = sitk.JoinSeries(*masks)

            # If it is a 3D CT, we just drop the list
            elif ct.cube_hu.GetDimension() == 3:
                masks = masks[0]
            else:
                raise ValueError("Sanity Check failed -- unsupported CT dimensionality")

            # Check Objectives
            if len(vdata) > 5:
                objectives = vdata[5]
                if not isinstance(objectives, list):
                    objectives = [objectives]
            else:
                objectives = []

            voi = validate_voi(
                name=str(vdata[1]),
                voi_type=str(vdata[2]),
                mask=masks,
                ct_image=ct,
                objectives=objectives,
            )

            voi_list.append(voi)

        cst_dict = {"vois": voi_list, "ct_image": ct}
        return StructureSet.model_validate(cst_dict)

    raise ValueError("Invalid input data for creating a StructureSet.")


def validate_cst(
    cst_data: Union[dict[str, Any], StructureSet, None] = None,
    ct: Union[CT, dict, None] = None,
    **kwargs,
) -> StructureSet:
    """
    Validate StructureSet.

    Parameters
    ----------
    cst_data : Union[dict[str, Any], StructureSet, None] , optional
        The input data to create the CT object from. Can be a dictionary,
        existing CT object, file path, or None.

    ct : Union[CT,dict,None], CT, None], optional
        The input data to create the CT object from. Can be a dictionary,
        existing CT object, or None.

    **kwargs
        Additional keyword arguments to create the CT object.

    Returns
    -------
    cst
        A StructureSet object created from the input data or keyword arguments.
    """
    return create_cst(cst_data, ct, **kwargs)
