#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Filename: path_info
# @Date: 2025/6/21
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

import os


class PathInfo(object):
    ROOT_PATH = r'D:\Onedrive\Temp\Projects\GlassDoor'

    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    OUTPUT_PATH = os.path.join(ROOT_PATH, 'regression_data')
    TEMP_PATH = os.path.join(ROOT_PATH, 'temp')
