import os
import logging
import warnings

from nano_emox.common.registry import registry
from nano_emox.datasets.datasets.base_dataset import BaseDataset
from nano_emox.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from nano_emox.datasets.datasets.mer2025ov_dataset import MER2025OV_Dataset
from nano_emox.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
from nano_emox.datasets.datasets.ovmerd_dataset import OVMERD_Dataset
from nano_emox.datasets.datasets.mer2023 import MER2023_Dataset
from nano_emox.datasets.datasets.mer2024 import MER2024_Dataset
from nano_emox.datasets.datasets.meld import MELD_Dataset
from nano_emox.datasets.datasets.meld_train import MELD_Train_Dataset
from nano_emox.datasets.datasets.cmumosi  import CMUMOSI_Dataset
from nano_emox.datasets.datasets.cmumosei import CMUMOSEI_Dataset
from nano_emox.datasets.datasets.sims import SIMS_Dataset
from nano_emox.datasets.datasets.simsv2 import SIMSv2_Dataset
from nano_emox.datasets.datasets.iemocap import IEMOCAPFour_Dataset
from nano_emox.datasets.datasets.iemocap_train import IEMOCAP_Train_Dataset
from nano_emox.datasets.datasets.avamerg_dataset import AvaMERG_Dataset
from nano_emox.datasets.datasets.merr_fine_dataset import MERRFine_Dataset
from nano_emox.datasets.datasets.emer import EMER_Dataset
from nano_emox.datasets.datasets.caer_ferv39k_dataset import CAER_FERV39K_Dataset
from nano_emox.datasets.datasets.m3ed_dataset import M3ED_Dataset
from nano_emox.datasets.datasets.crema_dataset import CREMA_Dataset
from nano_emox.datasets.datasets.mintrec2_dataset import MIntRec2_Dataset
from nano_emox.datasets.datasets.mintrec_dataset import MIntRec_Dataset

@registry.register_builder("mer2023")
class MER2023Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2023_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2023")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("mer2024")
class MER2024Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2024_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2024")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets



@registry.register_builder("meld")
class MELDBuilder(BaseDatasetBuilder):
    train_dataset_cls = MELD_Dataset

    def build_datasets(self):
        logging.info("Building datasets MELD")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("iemocapfour")
class IEMOCAPFourBuilder(BaseDatasetBuilder):
    train_dataset_cls = IEMOCAPFour_Dataset

    def build_datasets(self):
        logging.info("Building datasets IEMOCAPFour")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    

@registry.register_builder("cmumosi")
class CMUMOSIBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMUMOSI_Dataset

    def build_datasets(self):
        logging.info("Building datasets CMUMOSI")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets



@registry.register_builder("cmumosei")
class CMUMOSEIBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMUMOSEI_Dataset

    def build_datasets(self):
        logging.info("Building datasets CMUMOSEI")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("sims")
class SIMSBuilder(BaseDatasetBuilder):
    train_dataset_cls = SIMS_Dataset

    def build_datasets(self):
        logging.info("Building datasets SIMS")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("simsv2")
class SIMSv2Builder(BaseDatasetBuilder):
    train_dataset_cls = SIMSv2_Dataset

    def build_datasets(self):
        logging.info("Building datasets SIMSv2")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("mer2025ov")
class MER2025OV_Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2025OV_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2025OV_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    

@registry.register_builder("mercaptionplus")
class MERCaptionPlus_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERCaptionPlus_Dataset

    def build_datasets(self):
        logging.info("Building datasets MERCaptionPlus_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    

@registry.register_builder("ovmerd")
class OVMERD_Builder(BaseDatasetBuilder):
    train_dataset_cls = OVMERD_Dataset

    def build_datasets(self):
        logging.info("Building datasets OVMERD_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    
@registry.register_builder("avamerg")
class AvaMERG_Builder(BaseDatasetBuilder):
    train_dataset_cls = AvaMERG_Dataset

    def build_datasets(self):
        logging.info("Building datasets AvaMERG_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("meld_train")
class MELD_Train_Builder(BaseDatasetBuilder):
    train_dataset_cls = MELD_Train_Dataset

    def build_datasets(self):
        logging.info("Building datasets MELD_Train")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
        
@registry.register_builder("merrfine")
class MERRFine_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERRFine_Dataset

    def build_datasets(self):
        logging.info("Building datasets MERRFine")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("iemocap_train")
class IEMOCAP_Train_Builder(BaseDatasetBuilder):
    train_dataset_cls = IEMOCAP_Train_Dataset

    def build_datasets(self):
        logging.info("Building datasets IEMOCAP_Train")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets

@registry.register_builder("emer")
class EMER_Builder(BaseDatasetBuilder):
    train_dataset_cls = EMER_Dataset

    def build_datasets(self):
        logging.info("Building datasets IEMOCAP_Train")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets

@registry.register_builder("m3ed")
class M3ED_Builder(BaseDatasetBuilder):
    train_dataset_cls = M3ED_Dataset

    def build_datasets(self):
        logging.info("Building datasets M3ED")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets

@registry.register_builder("crema")
class CREMA_Builder(BaseDatasetBuilder):
    train_dataset_cls = CREMA_Dataset

    def build_datasets(self):
        logging.info("Building datasets CREMA")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets

@registry.register_builder("caer_ferv39k")
class CAER_FERV39K_Builder(BaseDatasetBuilder):
    train_dataset_cls = CAER_FERV39K_Dataset

    def build_datasets(self):
        logging.info("Building datasets CAER_FERV39K")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets

@registry.register_builder("mintrec")
class MIntRec_Builder(BaseDatasetBuilder):
    train_dataset_cls = MIntRec_Dataset

    def build_datasets(self):
        logging.info("Building datasets MIntRec")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets

@registry.register_builder("mintrec2")
class MIntRec2_Builder(BaseDatasetBuilder):
    train_dataset_cls = MIntRec2_Dataset

    def build_datasets(self):
        logging.info("Building datasets MIntRec2")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets