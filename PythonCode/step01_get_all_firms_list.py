#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Filename: step01_get_all_firms_list
# @Date: 2025/6/22
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

import os

import pandas as pd

from Constants import Constants as const


if __name__ == '__main__':
    # 读取数据文件
    data = pd.read_stata(os.path.join(r'D:\Users\wangy\PycharmProjects\GlassDoor\Reference', "Russell3000.dta"))

    # 处理gvkey列：去除字母并转换为整数类型
    data['gvkey'] = data['gvkey'].str.replace(r'[^0-9]', '', regex=True).astype(int)

    # 保存处理后的数据
    output_path = os.path.join(const.TEMP_PATH, "russell3000_glassdoor_page.pkl")
    data.to_pickle(output_path)

    ue_df = pd.read_stata(os.path.join(const.DATA_PATH, 'union_election_match_gvkey.dta'))
    ue_gvkey_list = ue_df.loc[ue_df[const.YEAR] >= 2008, const.GVKEY].dropna().drop_duplicates()

    ue_with_web = ue_gvkey_list.to_frame.merge(data, on=[const.GVKEY], how='left')
    ue_with_web.to_excel(os.path.join(const.TEMP_PATH, '20250622_ue_with_glassdoor_web.xlsx'), index=False)
