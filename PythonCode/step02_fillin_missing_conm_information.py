#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Filename: step02_fillin_missing_conm_information
# @Date: 2025/6/22
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
GVKEY: 4828 to 23667
"""

import os

import pandas as pd

from Constants import Constants as const

if __name__ == '__main__':
    ue_with_web = pd.read_excel(os.path.join(const.TEMP_PATH, '20250622_ue_with_glassdoor_web.xlsx'))
    ctat_df = pd.read_csv(os.path.join(const.COMPUSTAT_PATH, '1985_2024_ctat_firm_names.zip'))
    # missing_index = ue_with_web[ue_with_web[const.GVKEY].isnull()].index
    #
    # for i in missing_index:
    #     gvkey = ue_with_web.loc[i, const.GVKEY]
    #     tmp_ctat_name = ctat_df[ctat_df[const.GVKEY] == gvkey, 'conm'].dropna()[-1]
    #     ue_with_web.loc[i, 'conm'] = tmp_ctat_name

    company_name_map = ctat_df[['gvkey', 'conm']].dropna(how='any').drop_duplicates().set_index('gvkey')['conm']

    # 4. 填充缺失的公司名称
    # 方法1: 使用map函数直接填充
    ue_with_web['conm'] = ue_with_web['conm'].fillna(
        ue_with_web[const.GVKEY].map(company_name_map)
    )

    ue_with_web.to_excel(os.path.join(const.TEMP_PATH, '20250622_ue_with_glassdoor_web_fill_miss.xlsx'), index=False)
