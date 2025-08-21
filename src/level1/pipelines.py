from typing import Any

from src.core.registry import Registry
from src.level1.transforms import (
    AddToroidalAngle2,
    DropCoordinates,
    DropDatasets,
    DropErrors,
    DropZeroDataset,
    DropZeroDimensions,
    InterpolateAxis,
    LCFSTransform,
    Level1UDAGeometryLoader,
    MapDict,
    MergeDatasets,
    ProcessImage,
    RenameDimensions,
    RenameVariables,
    ReplaceInvalidValues,
    TensoriseChannels,
    TransformUnits,
)


class Pipeline:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class Pipelines:
    @property
    def group_mapping_file(self):
        raise NotImplementedError()

    @property
    def dimension_mapping_file(self):
        raise NotImplementedError()

    def get(self, name: str) -> Pipeline:
        if name not in self.pipelines:
            raise RuntimeError(f"{name} is not a registered source!")
        return self.pipelines[name]


class MASTUPipelines(Pipelines):
    @property
    def group_mapping_file(self):
        return "mappings/level1/mastu/groups.json"

    @property
    def dimension_mapping_file(self):
        return "mappings/level1/mastu/dimensions.json"

    @property
    def variable_mapping_file(self):
        return "mappings/level1/mastu/variables.json"

    def __init__(self) -> None:
        self.pipelines = {
            "abm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    DropDatasets(["cel3_bg_z", "cel3_radial_z", "cel3_ss_z"]),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aga": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ahx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ait": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aiv": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "alp": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ams": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    DropErrors(["stokes_s3", "stokes_s2", "stokes_s1"]),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ane": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anu": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_ch(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_ch(\d+)"),
                    TensoriseChannels("tcam", regex=r"tcam_ch(\d+)"),
                    TensoriseChannels("vcam", regex=r"vcam_ch(\d+)"),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayd": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "epm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    DropDatasets(
                        [
                            "input_numericalcontrols_ww_knotdim",
                            "input_numericalcontrols_ne_knotdim",
                            "input_numericalcontrols_pp_knotdim",
                            "input_numericalcontrols_ffp_knotdim",
                            "input_limiter_unitydim",
                            "input_constraints_pfcircuits_shortname",
                            "input_constraints_pfcircuits_pfcircuitsdim",
                            "input_constraints_magneticprobes_shortname",
                            "input_constraints_magneticprobes_magneticprobedim",
                            "input_constraints_fluxloops_strdim_shortname",
                            "input_constraints_fluxloops_fluxloopdim",
                        ]
                    ),
                    DropCoordinates("output_radialprofiles_totalpressure", ["r"]),
                    DropCoordinates("output_radialprofiles_toroidalflux", ["r"]),
                    DropCoordinates("output_radialprofiles_staticpressure", ["r"]),
                    DropCoordinates("output_radialprofiles_staticpprime", ["r"]),
                    DropCoordinates("output_radialprofiles_rotationalpressure", ["r"]),
                    DropCoordinates("output_radialprofiles_radialcoord", ["r"]),
                    DropCoordinates("output_radialprofiles_r", ["r"]),
                    DropCoordinates("output_radialprofiles_q", ["r"]),
                    DropCoordinates("output_radialprofiles_poloidalarea", ["r"]),
                    DropCoordinates("output_radialprofiles_plasmavolume", ["r"]),
                    DropCoordinates("output_radialprofiles_plasmadensity", ["r"]),
                    DropCoordinates(
                        "output_radialprofiles_normalizedtoroidalflux", ["r"]
                    ),
                    DropCoordinates(
                        "output_radialprofiles_normalizedpoloidalflux", ["r"]
                    ),
                    DropCoordinates("output_radialprofiles_jphi", ["r"]),
                    DropCoordinates("output_radialprofiles_ffprime", ["r"]),
                    DropCoordinates("output_radialprofiles_bz", ["r"]),
                    DropCoordinates("output_radialprofiles_bt", ["r"]),
                    DropCoordinates("output_radialprofiles_br", ["r"]),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "epq": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    DropDatasets(
                        [
                            "input_numericalcontrols_ww_knotdim",
                            "input_numericalcontrols_ne_knotdim",
                            "input_numericalcontrols_pp_knotdim",
                            "input_numericalcontrols_ffp_knotdim",
                            "input_limiter_unitydim",
                            "input_constraints_pfcircuits_shortname",
                            "input_constraints_pfcircuits_pfcircuitsdim",
                            "input_constraints_magneticprobes_shortname",
                            "input_constraints_magneticprobes_magneticprobedim",
                            "input_constraints_fluxloops_strdim_shortname",
                            "input_constraints_fluxloops_fluxloopdim",
                        ]
                    ),
                    DropCoordinates("output_radialprofiles_totalpressure", ["r"]),
                    DropCoordinates("output_radialprofiles_toroidalflux", ["r"]),
                    DropCoordinates("output_radialprofiles_staticpressure", ["r"]),
                    DropCoordinates("output_radialprofiles_staticpprime", ["r"]),
                    DropCoordinates("output_radialprofiles_rotationalpressure", ["r"]),
                    DropCoordinates("output_radialprofiles_radialcoord", ["r"]),
                    DropCoordinates("output_radialprofiles_r", ["r"]),
                    DropCoordinates("output_radialprofiles_q", ["r"]),
                    DropCoordinates("output_radialprofiles_poloidalarea", ["r"]),
                    DropCoordinates("output_radialprofiles_plasmavolume", ["r"]),
                    DropCoordinates("output_radialprofiles_plasmadensity", ["r"]),
                    DropCoordinates(
                        "output_radialprofiles_normalizedtoroidalflux", ["r"]
                    ),
                    DropCoordinates(
                        "output_radialprofiles_normalizedpoloidalflux", ["r"]
                    ),
                    DropCoordinates("output_radialprofiles_jphi", ["r"]),
                    DropCoordinates("output_radialprofiles_ffprime", ["r"]),
                    DropCoordinates("output_radialprofiles_bz", ["r"]),
                    DropCoordinates("output_radialprofiles_bt", ["r"]),
                    DropCoordinates("output_radialprofiles_br", ["r"]),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "rba": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rbb": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rbc": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rgb": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rgc": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "xbt": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("bes", regex=r"channel(\d+)"),
                ]
            ),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xim": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xma": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("rtdi_01", regex=r"rtdi_01_ch(\d+)$"),
                    TensoriseChannels("rtdi_02", regex=r"rtdi_02_ch(\d+)$"),
                    TensoriseChannels("rtdi_03", regex=r"rtdi_03_ch(\d+)$"),
                    TensoriseChannels("rtdi_04", regex=r"rtdi_04_ch(\d+)$"),
                    TensoriseChannels("rtdi_05", regex=r"rtdi_05_ch(\d+)$"),
                    TensoriseChannels("rtdi_06", regex=r"rtdi_06_ch(\d+)$"),
                    TensoriseChannels("rtdi_07", regex=r"rtdi_07_ch(\d+)$"),
                    TensoriseChannels("rtdi_08", regex=r"rtdi_08_ch(\d+)$"),
                    TensoriseChannels("sanx20_01", regex=r"sanx20_01_ch(\d+)$"),
                    TensoriseChannels("sanx20_02", regex=r"sanx20_02_ch(\d+)$"),
                    TensoriseChannels("sanx21_01", regex=r"sanx21_01_ch(\d+)$"),
                    TensoriseChannels("sanx21_02", regex=r"sanx21_02_ch(\d+)$"),
                ]
            ),
            "xmb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xmc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("acq216_202", regex=r"acq216_202_ch(\d+)"),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_ch(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_ch(\d+)"),
                    TensoriseChannels("tcam", regex=r"tcam_ch(\d+)"),
                    TensoriseChannels("vcam", regex=r"vcam_ch(\d+)"),
                ]
            ),
        }


class MASTPipelines(Pipelines):
    @property
    def group_mapping_file(self):
        return "mappings/level1/mast/groups.json"

    @property
    def dimension_mapping_file(self):
        return "mappings/level1/mast/dimensions.json"

    @property
    def variable_mapping_file(self):
        return "mappings/level1/mast/variables.json"

    def __init__(self) -> None:
        self.pipelines = {
            "abm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "acc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ada": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aga": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "adg": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ahx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aim": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "air": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ait": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "alp": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ama": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("ccbv"),
                    Level1UDAGeometryLoader('centrecolumn/t1', "ccbv", "/magnetics/pickup", "/common/uda-scratch/jg3176/pickup_coils.nc"),
                    AddToroidalAngle2("ccbv", "ccbv_channel"),

                    TensoriseChannels("obr"),
                    Level1UDAGeometryLoader('outervessel/t1/obr', "obr", "/magnetics/pickup", "/common/uda-scratch/jg3176/pickup_coils.nc"),
                    AddToroidalAngle2("obr", "obr_channel"),

                    TensoriseChannels("obv"),
                    Level1UDAGeometryLoader('outervessel/t1/obv', "obv", "/magnetics/pickup", "/common/uda-scratch/jg3176/pickup_coils.nc"),
                    AddToroidalAngle2("obv", "obv_channel"),

                    TensoriseChannels("fl_cc"),
                    Level1UDAGeometryLoader('centrecolumn', "fl_cc", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),
                    
                    TensoriseChannels("fl_p2l", regex=r"fl_p2l_(\d+)"),
                    Level1UDAGeometryLoader('p2/lower', "fl_p2l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p3l", regex=r"fl_p3l_(\d+)"),
                    Level1UDAGeometryLoader('p3/lower', "fl_p3l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p4l", regex=r"fl_p4l_(\d+)"),
                    Level1UDAGeometryLoader('p4/lower', "fl_p4l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p5l", regex=r"fl_p5l_(\d+)"),
                    Level1UDAGeometryLoader('p5/lower', "fl_p5l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p6l", regex=r"fl_p6l_(\d+)"),
                    Level1UDAGeometryLoader('p6/lower', "fl_p6l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p2u", regex=r"fl_p2u_(\d+)"),
                    Level1UDAGeometryLoader('p2/upper', "fl_p2u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p3u", regex=r"fl_p3u_(\d+)"),
                    Level1UDAGeometryLoader('p3/upper', "fl_p3u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p4u", regex=r"fl_p4u_(\d+)"),
                    Level1UDAGeometryLoader('p4/upper', "fl_p4u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p5u", regex=r"fl_p5u_(\d+)"),
                    Level1UDAGeometryLoader('p5/upper', "fl_p5u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    DropZeroDataset(),
                    TransformUnits(),
                    Level1UDAGeometryLoader('p2/p2_inner_upper', "p2_inner_upper", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p2/p2_inner_lower', "p2_inner_lower", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p2/p2_outer_upper', "p2_outer_upper", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p2/p2_outer_lower', "p2_outer_lower", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),

                    Level1UDAGeometryLoader('p3/p3_upper', "p3_upper", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p3/p3_lower', "p3_lower", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),

                    Level1UDAGeometryLoader('p4/p4_upper', "p4_upper", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p4/p4_lower', "p4_lower", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),

                    Level1UDAGeometryLoader('p5/p5_upper', "p5_upper", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p5/p5_lower', "p5_lower", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    
                    Level1UDAGeometryLoader('p6/p6_upper', "p6_upper", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                    Level1UDAGeometryLoader('p6/p6_lower', "p6_lower", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),

                    Level1UDAGeometryLoader('sol/sol', "sol", "/magnetics/pfcoil", "/common/uda-scratch/jg3176/pfcoils.nc"),
                ]
            ),
            "amh": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    Level1UDAGeometryLoader('centrecolumn/botcol', "botcol", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    Level1UDAGeometryLoader('centrecolumn/endcrown_l', "endcrown_l", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    Level1UDAGeometryLoader('centrecolumn/endcrown_u', "endcrown_u", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("incon"),
                    Level1UDAGeometryLoader('centrecolumn/incon', "incon", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),
                    
                    TensoriseChannels("lhorw"),
                    Level1UDAGeometryLoader('vessel/lhorw', "lhorw", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("mid"),
                    Level1UDAGeometryLoader('vessel/mid', "mid", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("p2larm"),
                    Level1UDAGeometryLoader('p2/p2larm', "p2larm", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),
                
                    TensoriseChannels("p2ldivpl"),
                    Level1UDAGeometryLoader('p2/p2ldivpl', "p2ldivpl", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("p2uarm"),
                    Level1UDAGeometryLoader('p2/p2uarm', "p2uarm", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),
                    
                    TensoriseChannels("p2udivpl"),
                    Level1UDAGeometryLoader('p2/p2udivpl', "p2udivpl", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("ring"),
                    Level1UDAGeometryLoader('centrecolumn/ring', "ring", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("rodgr"),
                    Level1UDAGeometryLoader('centrecolumn/rodgr', "rodgr", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    Level1UDAGeometryLoader('centrecolumn/topcol', "topcol", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("uhorw"),
                    Level1UDAGeometryLoader('vessel/uhorw', "uhorw", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    TensoriseChannels("vertw"),
                    Level1UDAGeometryLoader('vessel/vertw', "vertw", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),

                    Level1UDAGeometryLoader('cases', "coil_case", "/passive/efit", "/common/uda-scratch/jg3176/passivestructures.nc"),
                ]
            ),
            "ams": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ane": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ant": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anu": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aoe": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "arp": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TensoriseChannels("sad_m"),
                    TransformUnits(),
                ]
            ),
            "asx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "atm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    DropCoordinates("segment_number", ["time_segment"]),
                    DropCoordinates("angle", ["radial_index"]),
                    DropCoordinates("polyname", ["radial_index"]),
                    DropCoordinates("pulse", ["radial_index"]),
                    DropCoordinates("scat_length", ["radial_index"]),
                    DropDatasets(["time"]),
                    MergeDatasets(),
                    InterpolateAxis("time", "zero"),
                    TransformUnits(),
                ]
            ),
            "aye": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    DropCoordinates("segment_number", ["time_segment"]),
                    DropDatasets(["time"]),
                    MergeDatasets(),
                    InterpolateAxis("time", "zero"),
                    TransformUnits(),
                ]
            ),
            "efm": Pipeline(
                [
                    DropDatasets(
                        [
                            "all_times",
                            "fcoil_n",
                            "fcoil_segs_n",
                            "limitern",
                            "magpr_n",
                            "silop_n",
                            "shot_number",
                            "time",
                        ]
                    ),
                    MapDict(ReplaceInvalidValues()),
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    LCFSTransform(),
                    TransformUnits(),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "rba": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rbb": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rbc": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rcc": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rca": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rco": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rdd": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rgb": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rgc": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rir": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rit": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "rzz": Pipeline(
                [MapDict(RenameVariables(self.variable_mapping_file)), ProcessImage()]
            ),
            "xbt": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("bes", regex=r"channel(\d+)"),
                ]
            ),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xim": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xmo": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xpc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_(\d+)"),
                    Level1UDAGeometryLoader('lower_horizontal', "hcam_l", "/xraycams/core", "/common/uda-scratch/jg3176/xraycams.nc"),

                    TensoriseChannels("hcam_u", regex=r"hcam_u_(\d+)"),
                    Level1UDAGeometryLoader('upper_horizontal', "hcam_u", "/xraycams/core", "/common/uda-scratch/jg3176/xraycams.nc"),

                    Level1UDAGeometryLoader('third_horizontal', "hcam_third", "/xraycams/core", "/common/uda-scratch/jg3176/xraycams.nc"),

                    Level1UDAGeometryLoader('inner_vertical', "vcam_i", "/xraycams/core", "/common/uda-scratch/jg3176/xraycams.nc"),

                    Level1UDAGeometryLoader('outer_vertical', "vcam_o", "/xraycams/core", "/common/uda-scratch/jg3176/xraycams.nc"),

                    TensoriseChannels("tcam", regex=r"tcam_(\d+)"),
                    Level1UDAGeometryLoader("tangential", "tcam", "/xraycams/core", "/common/uda-scratch/jg3176/xraycams.nc"),
                ]
            ),
            "xma": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("ccbv", regex=r"ccbv_(\d+)"),
                    Level1UDAGeometryLoader('centrecolumn/t1', "ccbv", "/magnetics/pickup", "/common/uda-scratch/jg3176/pickup_coils.nc"),
                    AddToroidalAngle2("ccbv", "ccbv_channel"),

                    TensoriseChannels("obr", regex=r"obr_(\d+)"),
                    Level1UDAGeometryLoader('outervessel/t1/obr', "obr", "/magnetics/pickup", "/common/uda-scratch/jg3176/pickup_coils.nc"),
                    AddToroidalAngle2("obr", "obr_channel"),

                    TensoriseChannels("obv", regex=r"obv_(\d+)"),
                    Level1UDAGeometryLoader('outervessel/t1/obv', "obv", "/magnetics/pickup", "/common/uda-scratch/jg3176/pickup_coils.nc"),
                    AddToroidalAngle2("obv", "obv_channel"),

                    TensoriseChannels("fl_cc"),
                    Level1UDAGeometryLoader('centrecolumn', "fl_cc", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),
                    
                    TensoriseChannels("fl_p2l", regex=r"fl_p2l_(\d+)"),
                    Level1UDAGeometryLoader('p2/lower', "fl_p2l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p3l", regex=r"fl_p3l(\d+)"),
                    Level1UDAGeometryLoader('p3/lower', "fl_p3l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p4l", regex=r"fl_p4l(\d+)"),
                    Level1UDAGeometryLoader('p4/lower', "fl_p4l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p5l", regex=r"fl_p5l(\d+)"),
                    Level1UDAGeometryLoader('p5/lower', "fl_p5l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p6l", regex=r"fl_p6l(\d+)"),
                    Level1UDAGeometryLoader('p6/lower', "fl_p6l", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p2u", regex=r"fl_p2u(\d+)"),
                    Level1UDAGeometryLoader('p2/upper', "fl_p2u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p3u", regex=r"fl_p3u(\d+)"),
                    Level1UDAGeometryLoader('p3/upper', "fl_p3u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p4u", regex=r"fl_p4u(\d+)"),
                    Level1UDAGeometryLoader('p4/upper', "fl_p4u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),

                    TensoriseChannels("fl_p5u", regex=r"fl_p5u(\d+)"),
                    Level1UDAGeometryLoader('p5/upper', "fl_p5u", "/magnetics/fluxloops", "/common/uda-scratch/jg3176/fluxloops.nc"),
                ]
            ),
            "xmb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("sad_out_l"),
                    Level1UDAGeometryLoader('lower', "sad_out_l", "/magnetics/saddlecoils", "/common/uda-scratch/jg3176/saddle.nc"),
                    TensoriseChannels("sad_out_m"),
                    Level1UDAGeometryLoader('middle', "sad_out_m", "/magnetics/saddlecoils", "/common/uda-scratch/jg3176/saddle.nc"),
                    TensoriseChannels("sad_out_u"),
                    Level1UDAGeometryLoader('upper', "sad_out_u", "/magnetics/saddlecoils", "/common/uda-scratch/jg3176/saddle.nc"),
                ]
            ),
            "xmc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("cc_mt", regex=r"cc_mt_(\d+)"),
                    Level1UDAGeometryLoader('centrecolumn/toroidal', "cc_mt", "/magnetics/mirnov", "/common/uda-scratch/jg3176/mirnovs.nc"),
                    TensoriseChannels("cc_mv", regex=r"cc_mv_(\d+)"),
                    Level1UDAGeometryLoader('centrecolumn/vertical', "cc_mv", "/magnetics/mirnov", "/common/uda-scratch/jg3176/mirnovs.nc"),
                    TensoriseChannels("omv", regex=r"omv_(\d+)"),
                    Level1UDAGeometryLoader('outervessel/vertical', "omv", "/magnetics/mirnov", "/common/uda-scratch/jg3176/mirnovs.nc"),
                ]
            ),
            "xmp": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xms": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MapDict(DropZeroDimensions()),
                    MapDict(DropZeroDataset()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("channels", regex=r"ch(\d+)"),
                ]
            ),
        }


pipelines_registry = Registry[Pipelines]()
pipelines_registry.register("MAST", MASTPipelines)
pipelines_registry.register("MASTU", MASTUPipelines)
