from typing import Any

from src.registry import Registry
from src.transforms import (
    AddGeometry,
    AlignChannels,
    DropCoordinates,
    DropDatasets,
    DropZeroDataset,
    DropZeroDimensions,
    InterpolateAxis,
    LCFSTransform,
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
        return "mappings/mastu/groups.json"

    @property
    def dimension_mapping_file(self):
        return "mappings/mastu/dimensions.json"

    @property
    def variable_mapping_file(self):
        return "mappings/mastu/variables.json"

    def __init__(self) -> None:
        self.pipelines = {
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(self.variable_mapping_file),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "acu": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayd": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "epm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_ch(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_ch(\d+)"),
                    TensoriseChannels("tcam", regex=r"tcam_ch(\d+)"),
                ]
            ),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions(self.dimension_mapping_file)),
                    MapDict(RenameVariables(self.variable_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
        }


class MASTPipelines(Pipelines):
    @property
    def group_mapping_file(self):
        return "mappings/mast/groups.json"

    @property
    def dimension_mapping_file(self):
        return "mappings/mast/dimensions.json"

    @property
    def variable_mapping_file(self):
        return "mappings/mast/variables.json"

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
                    AddGeometry("ccbv", "geometry/data/amb/ccbv.parquet"),
                    AlignChannels("ccbv"),
                    TensoriseChannels("fl_cc"),
                    AddGeometry("fl_cc", "geometry/data/amb/fl_cc.parquet"),
                    AlignChannels("fl_cc"),
                    TensoriseChannels("fl_p2l", regex=r"fl_p2l_(\d+)"),
                    AddGeometry("fl_p2l", "geometry/data/amb/fl_p2l.parquet"),
                    AlignChannels("fl_p2l"),
                    TensoriseChannels("fl_p3l", regex=r"fl_p3l_(\d+)"),
                    AddGeometry("fl_p3l", "geometry/data/amb/fl_p3l.parquet"),
                    AlignChannels("fl_p3l"),
                    TensoriseChannels("fl_p4l", regex=r"fl_p4l_(\d+)"),
                    AddGeometry("fl_p4l", "geometry/data/amb/fl_p4l.parquet"),
                    AlignChannels("fl_p4l"),
                    TensoriseChannels("fl_p5l", regex=r"fl_p5l_(\d+)"),
                    AddGeometry("fl_p5l", "geometry/data/amb/fl_p5l.parquet"),
                    AlignChannels("fl_p5l"),
                    TensoriseChannels("fl_p6l", regex=r"fl_p6l_(\d+)"),
                    AddGeometry("fl_p6l", "geometry/data/amb/fl_p6l.parquet"),
                    AlignChannels("fl_p6l"),
                    TensoriseChannels("fl_p2u", regex=r"fl_p2u_(\d+)"),
                    AddGeometry("fl_p2u", "geometry/data/amb/fl_p2u.parquet"),
                    AlignChannels("fl_p2u"),
                    TensoriseChannels("fl_p3u", regex=r"fl_p3u_(\d+)"),
                    AddGeometry("fl_p3u", "geometry/data/amb/fl_p3u.parquet"),
                    AlignChannels("fl_p3u"),
                    TensoriseChannels("fl_p4u", regex=r"fl_p4u_(\d+)"),
                    AddGeometry("fl_p4u", "geometry/data/amb/fl_p4u.parquet"),
                    AlignChannels("fl_p4u"),
                    TensoriseChannels("fl_p5u", regex=r"fl_p5u_(\d+)"),
                    AddGeometry("fl_p5u", "geometry/data/amb/fl_p5u.parquet"),
                    AlignChannels("fl_p5u"),
                    TensoriseChannels("obr"),
                    AddGeometry("obr", "geometry/data/amb/xma_obr.parquet"),
                    AlignChannels("obr"),
                    TensoriseChannels("obv"),
                    AddGeometry("obv", "geometry/data/amb/xma_obv.parquet"),
                    AlignChannels("obv"),
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
                    AddGeometry(
                        "p2il_coil_current",
                        "geometry/data/amc/amc_p2il_coil_current.parquet",
                    ),
                    AlignChannels("p2il_coil_current"),
                    AddGeometry(
                        "p2iu_coil_current",
                        "geometry/data/amc/amc_p2iu_coil_current.parquet",
                    ),
                    AlignChannels("p2iu_coil_current"),
                    AddGeometry(
                        "p2l_case_current",
                        "geometry/data/amc/amc_p2l_case_current.parquet",
                    ),
                    AlignChannels("p2l_case_current"),
                    AddGeometry(
                        "p2ol_coil_current",
                        "geometry/data/amc/amc_p2ol_coil_current.parquet",
                    ),
                    AlignChannels("p2ol_coil_current"),
                    AddGeometry(
                        "p2ou_coil_current",
                        "geometry/data/amc/amc_p2ou_coil_current.parquet",
                    ),
                    AlignChannels("p2ou_coil_current"),
                    AddGeometry(
                        "p2u_case_current",
                        "geometry/data/amc/amc_p2u_case_current.parquet",
                    ),
                    AlignChannels("p2u_case_current"),
                    AddGeometry(
                        "p3l_case_current",
                        "geometry/data/amc/amc_p3l_case_current.parquet",
                    ),
                    AlignChannels("p3l_case_current"),
                    AddGeometry(
                        "p3l_coil_current",
                        "geometry/data/amc/amc_p3l_coil_current.parquet",
                    ),
                    AlignChannels("p3l_coil_current"),
                    AddGeometry(
                        "p3u_case_current",
                        "geometry/data/amc/amc_p3u_case_current.parquet",
                    ),
                    AlignChannels("p3u_case_current"),
                    AddGeometry(
                        "p3u_coil_current",
                        "geometry/data/amc/amc_p3u_coil_current.parquet",
                    ),
                    AlignChannels("p3u_coil_current"),
                    AddGeometry(
                        "p4l_case_current",
                        "geometry/data/amc/amc_p4l_case_current.parquet",
                    ),
                    AlignChannels("p4l_case_current"),
                    AddGeometry(
                        "p4l_coil_current",
                        "geometry/data/amc/amc_p4l_coil_current.parquet",
                    ),
                    AlignChannels("p4l_coil_current"),
                    AddGeometry(
                        "p4u_case_current",
                        "geometry/data/amc/amc_p4u_case_current.parquet",
                    ),
                    AlignChannels("p4u_case_current"),
                    AddGeometry(
                        "p4u_coil_current",
                        "geometry/data/amc/amc_p4u_coil_current.parquet",
                    ),
                    AlignChannels("p4u_coil_current"),
                    AddGeometry(
                        "p5l_case_current",
                        "geometry/data/amc/amc_p5l_case_current.parquet",
                    ),
                    AlignChannels("p5l_case_current"),
                    AddGeometry(
                        "p5l_coil_current",
                        "geometry/data/amc/amc_p5l_coil_current.parquet",
                    ),
                    AlignChannels("p5l_coil_current"),
                    AddGeometry(
                        "p5u_case_current",
                        "geometry/data/amc/amc_p5u_case_current.parquet",
                    ),
                    AlignChannels("p5u_case_current"),
                    AddGeometry(
                        "p5u_coil_current",
                        "geometry/data/amc/amc_p5u_coil_current.parquet",
                    ),
                    AlignChannels("p5u_coil_current"),
                    AddGeometry(
                        "p6l_case_current",
                        "geometry/data/amc/amc_p6l_case_current.parquet",
                    ),
                    AlignChannels("p6l_case_current"),
                    AddGeometry(
                        "p6l_coil_current",
                        "geometry/data/amc/amc_p6l_coil_current.parquet",
                    ),
                    AlignChannels("p6l_coil_current"),
                    AddGeometry(
                        "p6u_case_current",
                        "geometry/data/amc/amc_p6u_case_current.parquet",
                    ),
                    AlignChannels("p6u_case_current"),
                    AddGeometry(
                        "p6u_coil_current",
                        "geometry/data/amc/amc_p6u_coil_current.parquet",
                    ),
                    AlignChannels("p6u_coil_current"),
                    AddGeometry(
                        "sol_current", "geometry/data/amc/amc_sol_current.parquet"
                    ),
                    AlignChannels("sol_current"),
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
                    AddGeometry("botcol", "geometry/data/amm/amm_botcol.parquet"),
                    AlignChannels("botcol"),
                    AddGeometry(
                        "endcrown_l", "geometry/data/amm/amm_endcrown_l.parquet"
                    ),
                    AlignChannels("endcrown_l"),
                    AddGeometry(
                        "endcrown_u", "geometry/data/amm/amm_endcrown_u.parquet"
                    ),
                    AlignChannels("endcrown_u"),
                    TensoriseChannels("incon"),
                    AddGeometry("incon", "geometry/data/amm/amm_incon.parquet"),
                    AlignChannels("incon"),
                    TensoriseChannels("lhorw"),
                    AddGeometry("lhorw", "geometry/data/amm/amm_lhorw.parquet"),
                    AlignChannels("lhorw"),
                    TensoriseChannels("mid"),
                    AddGeometry("mid", "geometry/data/amm/amm_mid.parquet"),
                    AlignChannels("mid"),
                    AddGeometry("p2larm1", "geometry/data/amm/amm_p2larm1.parquet"),
                    AlignChannels("p2larm1"),
                    AddGeometry("p2larm2", "geometry/data/amm/amm_p2larm2.parquet"),
                    AlignChannels("p2larm2"),
                    AddGeometry("p2larm3", "geometry/data/amm/amm_p2larm3.parquet"),
                    AlignChannels("p2larm3"),
                    AddGeometry("p2ldivpl1", "geometry/data/amm/amm_p2ldivpl1.parquet"),
                    AlignChannels("p2ldivpl1"),
                    AddGeometry("p2ldivpl2", "geometry/data/amm/amm_p2ldivpl2.parquet"),
                    AlignChannels("p2ldivpl2"),
                    AddGeometry("p2uarm1", "geometry/data/amm/amm_p2uarm1.parquet"),
                    AlignChannels("p2uarm1"),
                    AddGeometry("p2uarm2", "geometry/data/amm/amm_p2uarm2.parquet"),
                    AlignChannels("p2uarm2"),
                    AddGeometry("p2uarm3", "geometry/data/amm/amm_p2uarm3.parquet"),
                    AlignChannels("p2uarm3"),
                    AddGeometry("p2udivpl1", "geometry/data/amm/amm_p2udivpl1.parquet"),
                    AlignChannels("p2udivpl1"),
                    TensoriseChannels("ring"),
                    AddGeometry("ring", "geometry/data/amm/amm_ring.parquet"),
                    AlignChannels("ring"),
                    TensoriseChannels("rodgr"),
                    AddGeometry("rodgr", "geometry/data/amm/amm_rodr.parquet"),
                    AlignChannels("rodgr"),
                    AddGeometry("topcol", "geometry/data/amm/amm_topcol.parquet"),
                    AlignChannels("topcol"),
                    TensoriseChannels("uhorw"),
                    AddGeometry("uhorw", "geometry/data/amm/amm_uhorw.parquet"),
                    AlignChannels("uhorw"),
                    TensoriseChannels("vertw"),
                    AddGeometry("vertw", "geometry/data/amm/amm_vertw.parquet"),
                    AlignChannels("vertw"),
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
                    TensoriseChannels("v_ste29", regex=r"v_ste29_(\d+)"),
                    AddGeometry(
                        "v_ste29", "geometry/data/xsx/ssx_inner_vertical_cam.parquet"
                    ),
                    AlignChannels("v_ste29"),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_(\d+)"),
                    AddGeometry(
                        "hcam_l", "geometry/data/xsx/ssx_lower_horizontal_cam.parquet"
                    ),
                    AlignChannels("hcam_l"),
                    TensoriseChannels("tcam", regex=r"tcam_(\d+)"),
                    AddGeometry("tcam", "geometry/data/xsx/ssx_tangential_cam.parquet"),
                    AlignChannels("tcam"),
                    TensoriseChannels("hpzr", regex=r"hpzr_(\d+)"),
                    AddGeometry(
                        "hpzr", "geometry/data/xsx/ssx_third_horizontal_cam.parquet"
                    ),
                    AlignChannels("hpzr"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_(\d+)"),
                    AddGeometry(
                        "hcam_u", "geometry/data/xsx/ssx_upper_horizontal_cam.parquet"
                    ),
                    AlignChannels("hcam_u"),
                    TensoriseChannels("v_ste36", regex=r"v_ste36_(\d+)"),
                    AddGeometry(
                        "v_ste36", "geometry/data/xsx/ssx_outer_vertical_cam.parquet"
                    ),
                    AlignChannels("v_ste36"),
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
                    TensoriseChannels("ccbv", regex=r"ccbv(\d+)"),
                    AddGeometry("ccbv", "geometry/data/xma/ccbv.parquet"),
                    AlignChannels("ccbv"),
                    TensoriseChannels("fl_cc"),
                    AddGeometry("fl_cc", "geometry/data/xma/fl_cc.parquet"),
                    AlignChannels("fl_cc"),
                    TensoriseChannels("fl_p2l"),
                    AddGeometry("fl_p2l", "geometry/data/xma/fl_p2l.parquet"),
                    AlignChannels("fl_p2l"),
                    TensoriseChannels("fl_p3l"),
                    AddGeometry("fl_p3l", "geometry/data/xma/fl_p3l.parquet"),
                    AlignChannels("fl_p3l"),
                    TensoriseChannels("fl_p4l"),
                    AddGeometry("fl_p4l", "geometry/data/xma/fl_p4l.parquet"),
                    AlignChannels("fl_p4l"),
                    TensoriseChannels("fl_p5l"),
                    AddGeometry("fl_p5l", "geometry/data/xma/fl_p5l.parquet"),
                    AlignChannels("fl_p5l"),
                    TensoriseChannels("fl_p6l"),
                    AddGeometry("fl_p6l", "geometry/data/xma/fl_p6l.parquet"),
                    AlignChannels("fl_p6l"),
                    TensoriseChannels("fl_p2u"),
                    AddGeometry("fl_p2u", "geometry/data/xma/fl_p2u.parquet"),
                    AlignChannels("fl_p2u"),
                    TensoriseChannels("fl_p3u"),
                    AddGeometry("fl_p3u", "geometry/data/xma/fl_p3u.parquet"),
                    AlignChannels("fl_p3u"),
                    TensoriseChannels("fl_p4u"),
                    AddGeometry("fl_p4u", "geometry/data/xma/fl_p4u.parquet"),
                    AlignChannels("fl_p4u"),
                    TensoriseChannels("fl_p5u"),
                    AddGeometry("fl_p5u", "geometry/data/xma/fl_p5u.parquet"),
                    AlignChannels("fl_p5u"),
                    TensoriseChannels("obr", regex=r"obr_(\d+)"),
                    TensoriseChannels("obr", regex=r"obr(\d+)"),
                    AddGeometry("obr", "geometry/data/xma/xma_obr.parquet"),
                    AlignChannels("obr"),
                    TensoriseChannels("obv", regex=r"obv_(\d+)"),
                    TensoriseChannels("obv", regex=r"obv(\d+)"),
                    AddGeometry("obv", "geometry/data/xma/xma_obv.parquet"),
                    AlignChannels("obv"),
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
                    AddGeometry("sad_out_l", "geometry/data/xmb/xmb_sad_l.parquet"),
                    AlignChannels("sad_out_l"),
                    TensoriseChannels("sad_out_u"),
                    AddGeometry("sad_out_u", "geometry/data/xmb/xmb_sad_u.parquet"),
                    AlignChannels("sad_out_u"),
                    TensoriseChannels("sad_out_m"),
                    AddGeometry("sad_out_m", "geometry/data/xmb/xmb_sad_m.parquet"),
                    AlignChannels("sad_out_m"),
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
                    AddGeometry("cc_mt", "geometry/data/xmc/ccmt.parquet"),
                    AlignChannels("cc_mt"),
                    TensoriseChannels("cc_mv", regex=r"cc_mv_(\d+)"),
                    AddGeometry("cc_mv", "geometry/data/xmc/ccmv.parquet"),
                    AlignChannels("cc_mv"),
                    TensoriseChannels("omv", regex=r"omv_(\d+)"),
                    AddGeometry("omv", "geometry/data/xmc/xmc_omv.parquet"),
                    AlignChannels("omv"),
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
