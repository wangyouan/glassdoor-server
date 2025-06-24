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

    ue_df = pd.read_pickle(os.path.join(const.TEMP_PATH, '1950_2025_union_election_gvkey3.pkl'))
    ue_gvkey_list = ue_df.loc[ue_df[const.ELECTION_YEAR] >= 2008, const.GVKEY].dropna().drop_duplicates()

    ue_with_web = ue_gvkey_list.to_frame().merge(data, on=[const.GVKEY], how='left')

    ue_with_web2 = pd.read_excel(os.path.join(const.TEMP_PATH, '20250622_ue_with_glassdoor_web_fill_miss.xlsx'))
    ue_with_web3 = ue_with_web.merge(ue_with_web2[[const.GVKEY, 'glassdoor_web']], on=[const.GVKEY], how='left')
    ue_with_web3['glassdoor_web'] = ue_with_web3['glassdoor_web_x'].fillna(ue_with_web3['glassdoor_web_y'])
    
    ue_with_web3.drop(['glassdoor_web_x', 'glassdoor_web_y'], axis=1).to_excel(
        os.path.join(const.TEMP_PATH, '20250624_ue_with_glassdoor_web.xlsx'), index=False)
