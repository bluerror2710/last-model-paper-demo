from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta


class PaperSafeStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    informative_timeframe = "4h"
    can_short = True
    startup_candle_count = 300

    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 48,
            "trade_limit": 2,
            "stop_duration_candles": 24,
            "only_per_pair": False,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 168,
            "trade_limit": 20,
            "stop_duration_candles": 48,
            "max_allowed_drawdown": 0.06,
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 4,
        },
    ]

    buy_ema_fast = IntParameter(8, 30, default=20, space="buy")
    buy_ema_slow = IntParameter(30, 100, default=50, space="buy")
    buy_rsi = IntParameter(45, 65, default=50, space="buy")
    buy_adx = IntParameter(15, 35, default=22, space="buy")
    buy_bbwidth = DecimalParameter(0.01, 0.08, decimals=3, default=0.02, space="buy")

    sell_ema_fast = IntParameter(8, 30, default=20, space="sell")
    sell_ema_slow = IntParameter(30, 100, default=50, space="sell")
    sell_rsi = IntParameter(35, 55, default=45, space="sell")

    minimal_roi = {
        "0": 0.324,
        "367": 0.201,
        "764": 0.046,
        "962": 0
    }
    stoploss = -0.029
    trailing_stop = True
    trailing_stop_positive = 0.216
    trailing_stop_positive_offset = 0.248
    trailing_only_offset_is_reached = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for p in range(8, 101):
            dataframe[f"ema_{p}"] = ta.EMA(dataframe, timeperiod=p)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        bb = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["close"]

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["hour"] = dataframe["date"].dt.hour

        if self.dp:
            inf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
            inf["ema_50"] = ta.EMA(inf, timeperiod=50)
            inf["ema_200"] = ta.EMA(inf, timeperiod=200)
            dataframe = merge_informative_pair(dataframe, inf[["date", "ema_50", "ema_200"]], self.timeframe, self.informative_timeframe, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ef = dataframe[f"ema_{self.buy_ema_fast.value}"]
        es = dataframe[f"ema_{self.buy_ema_slow.value}"]

        uptrend_4h = dataframe["ema_50_4h"] > dataframe["ema_200_4h"]
        downtrend_4h = dataframe["ema_50_4h"] < dataframe["ema_200_4h"]

        common_filters = (
            (dataframe["adx"] > self.buy_adx.value) &
            (dataframe["atr_pct"] > 0.003) &
            (dataframe["atr_pct"] < 0.03) &
            (dataframe["bb_width"] > self.buy_bbwidth.value) &
            (dataframe["hour"].between(7, 21)) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[
            common_filters & uptrend_4h & (ef > es) & (dataframe["rsi"] > self.buy_rsi.value) & (dataframe["macd"] > dataframe["macdsignal"]),
            "enter_long"
        ] = 1

        dataframe.loc[
            common_filters & downtrend_4h & (ef < es) & (dataframe["rsi"] < (100 - self.buy_rsi.value)) & (dataframe["macd"] < dataframe["macdsignal"]),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ef = dataframe[f"ema_{self.sell_ema_fast.value}"]
        es = dataframe[f"ema_{self.sell_ema_slow.value}"]

        dataframe.loc[
            ((ef < es) | (dataframe["rsi"] < self.sell_rsi.value) | (dataframe["adx"] < 14)) & (dataframe["volume"] > 0),
            "exit_long"
        ] = 1

        dataframe.loc[
            ((ef > es) | (dataframe["rsi"] > (100 - self.sell_rsi.value)) | (dataframe["adx"] < 14)) & (dataframe["volume"] > 0),
            "exit_short"
        ] = 1

        return dataframe
