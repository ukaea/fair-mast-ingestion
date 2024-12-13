from typing import Any
from src.registry import Registry
from src.transforms import (
    AddGeometry,
    AlignChannels,
    ASXTransform,
    DropCoordinates,
    DropDatasets,
    DropZeroDataset,
    DropZeroDimensions,
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
    def __init__(self) -> None:
        pass

    def get(self, name: str) -> Pipeline:
        if name not in self.pipelines:
            raise RuntimeError(f"{name} is not a registered source!")
        return self.pipelines[name]


class MASTUPipelines(Pipelines):
    def __init__(self) -> None:
        dim_mapping_file = "mappings/mastu/dimensions.json"

        self.pipelines = {
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "ip": "plasma_current",
                        }
                    ),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "acu": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayd": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "epm": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_ch(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_ch(\d+)"),
                    TensoriseChannels("tcam", regex=r"tcam_ch(\d+)"),
                ]
            ),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
        }


class MASTPipelines(Pipelines):
    def __init__(self) -> None:
        self.pipelines = {
            "abm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(DropZeroDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "acc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ada": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aga": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "adg": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ahx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aim": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "air": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ait": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "alp": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ama": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions()),
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
                    MapDict(RenameDimensions()),
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
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amm": Pipeline(
                [
                    MapDict(RenameDimensions()),
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
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ane": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ant": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anu": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aoe": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "arp": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asb": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TensoriseChannels("sad_m"),
                    TransformUnits(),
                ]
            ),
            "asx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(ASXTransform()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "atm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "r": "radius",
                        }
                    ),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    DropCoordinates("segment_number", ["time_segment"]),
                    DropDatasets(["time"]),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "r": "radius",
                        }
                    ),
                ]
            ),
            "aye": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "efm": Pipeline(
                [
                    DropDatasets(
                        [
                            "efm/fcoil_n",
                            "efm/fcoil_segs_n",
                            "efm/limitern",
                            "efm/magpr_n",
                            "efm/silop_n",
                            "efm/shot_number",
                        ]
                    ),
                    MapDict(ReplaceInvalidValues()),
                    MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    LCFSTransform(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "plasma_currc": "plasma_current_c",
                            "plasma_currx": "plasma_current_x",
                            "plasma_currrz": "plasma_current_rz",
                            "lcfsr_c": "lcfs_r",
                            "lcfsz_c": "lcfs_z",
                        }
                    ),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "rba": Pipeline([ProcessImage()]),
            "rbb": Pipeline([ProcessImage()]),
            "rbc": Pipeline([ProcessImage()]),
            "rcc": Pipeline([ProcessImage()]),
            "rca": Pipeline([ProcessImage()]),
            "rco": Pipeline([ProcessImage()]),
            "rdd": Pipeline([ProcessImage()]),
            "rgb": Pipeline([ProcessImage()]),
            "rgc": Pipeline([ProcessImage()]),
            "rir": Pipeline([ProcessImage()]),
            "rit": Pipeline([ProcessImage()]),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xim": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xmo": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xpc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    RenameVariables(
                        {
                            "hcaml#1": "hcam_l_1",
                            "hcaml#10": "hcam_l_10",
                            "hcaml#11": "hcam_l_11",
                            "hcaml#12": "hcam_l_12",
                            "hcaml#13": "hcam_l_13",
                            "hcaml#14": "hcam_l_14",
                            "hcaml#15": "hcam_l_15",
                            "hcaml#16": "hcam_l_16",
                            "hcaml#17": "hcam_l_17",
                            "hcaml#18": "hcam_l_18",
                            "hcaml#2": "hcam_l_2",
                            "hcaml#3": "hcam_l_3",
                            "hcaml#4": "hcam_l_4",
                            "hcaml#5": "hcam_l_5",
                            "hcaml#6": "hcam_l_6",
                            "hcaml#7": "hcam_l_7",
                            "hcaml#8": "hcam_l_8",
                            "hcaml#9": "hcam_l_9",
                            "hcamu#1": "hcam_u_1",
                            "hcamu#10": "hcam_u_10",
                            "hcamu#11": "hcam_u_11",
                            "hcamu#12": "hcam_u_12",
                            "hcamu#13": "hcam_u_13",
                            "hcamu#14": "hcam_u_14",
                            "hcamu#15": "hcam_u_15",
                            "hcamu#16": "hcam_u_16",
                            "hcamu#17": "hcam_u_17",
                            "hcamu#18": "hcam_u_18",
                            "hcamu#2": "hcam_u_2",
                            "hcamu#3": "hcam_u_3",
                            "hcamu#4": "hcam_u_4",
                            "hcamu#5": "hcam_u_5",
                            "hcamu#6": "hcam_u_6",
                            "hcamu#7": "hcam_u_7",
                            "hcamu#8": "hcam_u_8",
                            "hcamu#9": "hcam_u_9",
                            "tcam#1": "tcam_1",
                            "tcam#10": "tcam_10",
                            "tcam#11": "tcam_11",
                            "tcam#12": "tcam_12",
                            "tcam#13": "tcam_13",
                            "tcam#14": "tcam_14",
                            "tcam#15": "tcam_15",
                            "tcam#16": "tcam_16",
                            "tcam#17": "tcam_17",
                            "tcam#18": "tcam_18",
                            "tcam#2": "tcam_2",
                            "tcam#3": "tcam_3",
                            "tcam#4": "tcam_4",
                            "tcam#5": "tcam_5",
                            "tcam#6": "tcam_6",
                            "tcam#7": "tcam_7",
                            "tcam#8": "tcam_8",
                            "tcam#9": "tcam_9",
                        }
                    ),
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
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("ccbv", regex=r"ccbv_(\d+)"),
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
                    AddGeometry("obr", "geometry/data/xma/xma_obr.parquet"),
                    AlignChannels("obr"),
                    TensoriseChannels("obv", regex=r"obv_(\d+)"),
                    AddGeometry("obv", "geometry/data/xma/xma_obv.parquet"),
                    AlignChannels("obv"),
                ]
            ),
            "xmb": Pipeline(
                [
                    MapDict(RenameDimensions()),
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
                    MapDict(RenameDimensions()),
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
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xms": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
        }


pipelines_registry = Registry[Pipelines]()
pipelines_registry.register("MAST", MASTPipelines)
pipelines_registry.register("MASTU", MASTUPipelines)
