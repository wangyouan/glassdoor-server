#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Filename: __init__.py
# @Date: 2025/6/21
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

from .path_info import PathInfo


class Constants(PathInfo):
    GVKEY = 'gvkey'
    YEAR = 'year'
    SIC = 'sic'
    SIC2 = 'sic2'
    CUSIP = 'cusip'
    CUSIP8 = 'cusip8'
    TICKER = 'tic'

    VOTE_SHARE = 'vote_share'
    ELECTION_TYPE = 'type'
    IS_WIN = 'is_win'
    IS_CLOSE_ELECTION = 'is_close'
    MARGIN_OF_VICTORY = 'margin_of_victory'
    ABS_MARGIN_SHARE = 'margin_share_abs'
    NUM_VOTE_FOR = 'votes_for'
    NUM_VOTE_AGAINST = 'votes_against'
    NUM_VOTES = 'number_of_votes'
    ELECTION_YEAR = 'year_electionheld'
    ELECTION_MONTH = 'month_electionheld'
    ELECTION_DAY = 'day_electionheld'