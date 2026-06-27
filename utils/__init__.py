# -*- coding: utf-8 -*-
# @File    : __init__.py
# @Project : Project-Beta (Alpha-02)

"""本模块提供了工具包的公共接口和统一导出。"""

from utils.data_loader import load_enriched_data, load_alpha_data, compute_vwap
from utils.feature_engine import build_cross_section_features, build_sequence_data, build_vwap_return_label
from utils.preprocess import preprocess_price_data
from utils.dataset import CrossSectionDataset, SequenceDataset, split_dataset
from utils.metrics import rank_ic, calc_ic_series, ic_summary, group_return

__all__ = [
    "load_enriched_data",
    "load_alpha_data",
    "compute_vwap",
    "build_cross_section_features",
    "build_sequence_data",
    "build_vwap_return_label",
    "preprocess_price_data",
    "CrossSectionDataset",
    "SequenceDataset",
    "split_dataset",
    "rank_ic",
    "calc_ic_series",
    "ic_summary",
    "group_return",
]
