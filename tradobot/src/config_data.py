from __future__ import  annotations

# stocks - tickers for 1Day frequency
TICKER_LIST_test = ['HA', 'WBA']
TICKER_LIST_1 = ['HA', 'WBA', 'INCY']
TICKER_LIST_2 = ['HA', 'WBA', 'INCY', 'BIDU']
TICKER_LIST_3 = ['HA', 'WBA', 'INCY', 'BIDU', 'TCOM']
TICKER_LIST_4 = ['HA', 'WBA', 'INCY', 'AAPL']
TICKER_LIST_5 = ['HA', 'WBA', 'INCY', 'AAPL', 'COST']
TICKER_LIST_6 = ['HA', 'WBA', 'INCY', 'BIDU', 'TCOM', 'AAPL', 'COST']
TICKER_LIST_7 = ['BIDU', 'TCOM', 'AAPL', 'COST']

TICKER_LIST_lists = [TICKER_LIST_1, TICKER_LIST_2, TICKER_LIST_3, TICKER_LIST_4,
                     TICKER_LIST_5, TICKER_LIST_6, TICKER_LIST_7]

# TICKER_LIST_1 = ['MSFT']
# TICKER_LIST_2 = ['AMZN']
# TICKER_LIST_3 = ['SPX']
# TICKER_LIST_4 = ['SPY']
#
# TICKER_LISTS_lists = [TICKER_LIST_1, TICKER_LIST_2, TICKER_LIST_3, TICKER_LIST_4]

#TICKER_LIST_lists = [TICKER_LIST_6]

# Used in article, training:  June 2014–December 2019
# Used in article, validation: January 2020–December 2021


# DATA DATES FOR 1DAY FREQUENCY ##########################
TRAIN_START_DATE = '2014-07-01'
TRAIN_END_DATE = '2019-12-31'
# Used in article: January 2020 – December 2021
TEST_START_DATE = '2020-01-01'
# TEST_END_DATE = '2021-12-31'

TEST_END_DATE = '2022-12-31'

# opening and closing prices every time interval
TIME_INTERVAL = '1D'

# TRADE_START_DATE = "2023-05-01"
# TRADE_END_DATE = "2023-06-30"

INDICATORS = ["macd",
              "boll_ub",
              "boll_lb",
              "rsi_30",
              "dx_30",
              "close_30_sma",
              "close_60_sma", ]




# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # define owbn timezone
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = 'AK50KJ2T4L9E9H638HKM' # your ALPACA_API_KEY
ALPACA_API_SECRET = 'b6rcgyNXCDS5G0fLUIxR9fWmUdx7m4M20OL1JIFX' # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = 'https://paper-api.alpaca.markets'  # alpaca url


