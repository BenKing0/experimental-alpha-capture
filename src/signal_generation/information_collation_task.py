from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.schemas import INPUT_FIELDS
from src.signal_generation.load_feed import FeedLoader

PICKLE_DIRECTORY = "notebooks/pickled_feeds"


class InformationCollationPipeline:
    
    def __init__(self, api_key: str, tickers: List[str]) -> None:
        self.api_key = api_key
        self.tickers = tickers

        self.signals_df = pd.DataFrame(columns=[
            INPUT_FIELDS.ticker,
            INPUT_FIELDS.industry,
            INPUT_FIELDS.trailing_pe,
            INPUT_FIELDS.forward_pe,
            INPUT_FIELDS.analyst_direction,
            INPUT_FIELDS.analyst_target_price,
            INPUT_FIELDS.sentiment_score,
            INPUT_FIELDS.sentiment_direction,
            INPUT_FIELDS.sentiment_conviction,
            INPUT_FIELDS.execution_time,
            INPUT_FIELDS.valid_period,
        ])
        self.signals_df[INPUT_FIELDS.ticker] = pd.Series(tickers)
        self.signals_df.set_index(INPUT_FIELDS.ticker, inplace=True)

        self.loader = FeedLoader(api_key, pickle_dir=PICKLE_DIRECTORY)

    def run(self) -> pd.DataFrame:
        self.process_sentiment()
        self.process_fundamentals()
        self.process_analytics()

    def process_analytics(self) -> None:
        feed_json = self.loader.load_window_analytics(
            self.tickers,
            n_month=12,
            calculations=["STDDEV"],
            window=20,
            interval="DAILY"
        )
        
        self.signals_df.reset_index(inplace=True)
        self.signals_df[INPUT_FIELDS.valid_period] = self.signals_df[INPUT_FIELDS.ticker].apply(
            lambda x: self._scrape_window_analytics_feed(feed_json, ticker=x, window_size=20)
        )
        self.signals_df.set_index(INPUT_FIELDS.ticker, inplace=True)

    def process_fundamentals(self) -> None:
        feed_json = self.loader.load_fundamentals(self.tickers)

        self.signals_df.reset_index(inplace=True)
        self.signals_df[[
            INPUT_FIELDS.industry,
            INPUT_FIELDS.trailing_pe,
            INPUT_FIELDS.forward_pe,
            INPUT_FIELDS.analyst_target_price,
            '_AnalystRatingStrongBuy', 
            '_AnalystRatingBuy', 
            '_AnalystRatingHold', 
            '_AnalystRatingSell', 
            '_AnalystRatingStrongSell',
        ]] = self.signals_df.apply(
            lambda row: self._scrape_fundamental_feed(feed_json, row=row),
            axis = 1,
        )
        self.signals_df.set_index(INPUT_FIELDS.ticker, inplace=True)

        self.signals_df[[
            INPUT_FIELDS.analyst_direction,
            INPUT_FIELDS.analyst_conviction,
        ]] = (
            (
                self.signals_df['_AnalystRatingStrongBuy'] * 2
                + self.signals_df['_AnalystRatingBuy'] * 1
                + self.signals_df['_AnalystRatingHold'] * 0
                + self.signals_df['_AnalystRatingSell'] * -1
                + self.signals_df['_AnalystRatingStrongSell'] * -2
            ) / (
                self.signals_df['_AnalystRatingStrongBuy']
                + self.signals_df['_AnalystRatingBuy']
                + self.signals_df['_AnalystRatingHold']
                + self.signals_df['_AnalystRatingSell']
                + self.signals_df['_AnalystRatingStrongSell']
            )
        ).apply(lambda x: self._convert_analyst_score(x))

        self.signals_df.drop(columns=[
            '_AnalystRatingStrongBuy', 
            '_AnalystRatingBuy', 
            '_AnalystRatingHold', 
            '_AnalystRatingSell', 
            '_AnalystRatingStrongSell',
        ], inplace=True)

    def process_sentiment(self) -> None:
        feed_json = self.loader.load_news_sentiment(self.tickers)
        
        self.signals_df["_news_occurences"] = self._scrape_sentiment_feed(feed_json)
        self.signals_df[["_all_sentiment_scores", "_all_relevance_scores"]] = self.signals_df.apply(
            lambda row: self._collect_sentiment_scores(feed_json, row),
            axis=1,
        )

        self.signals_df[INPUT_FIELDS.sentiment_score] = self._aggregate_scores(self.signals_df)
        self.signals_df[[INPUT_FIELDS.sentiment_direction, INPUT_FIELDS.sentiment_conviction]] = (
            self.signals_df[INPUT_FIELDS.sentiment_score]
            .dropna()
            .apply(self._convert_sentiment_score)
        )

        self.signals_df.drop(columns=[
            "_news_occurences",
            "_all_sentiment_scores",
            "_all_relevance_scores",
        ], inplace=True)

    @staticmethod
    def _scrape_fundamental_feed(feed_json: Dict[str, dict], row: pd.Series) -> pd.Series:
        ticker = row[INPUT_FIELDS.ticker]
        if feed_json.get(ticker, None):
            _dict = feed_json[ticker]
            return pd.Series(
                {
                    INPUT_FIELDS.industry: str(_dict["Industry"]),
                    INPUT_FIELDS.trailing_pe: str(_dict["TrailingPE"]),
                    INPUT_FIELDS.forward_pe: str(_dict["ForwardPE"]),
                    INPUT_FIELDS.analyst_target_price: float(_dict['AnalystTargetPrice']),
                    '_AnalystRatingStrongBuy': int(_dict['AnalystRatingStrongBuy']), 
                    '_AnalystRatingBuy': int(_dict['AnalystRatingBuy']),
                    '_AnalystRatingHold': int(_dict['AnalystRatingHold']),
                    '_AnalystRatingSell': int(_dict['AnalystRatingSell']),
                    '_AnalystRatingStrongSell': int(_dict['AnalystRatingStrongSell']),
                }
            )
        else: 
            return pd.Series(
                {
                    INPUT_FIELDS.industry: None,
                    INPUT_FIELDS.trailing_pe: None,
                    INPUT_FIELDS.forward_pe: None,
                    INPUT_FIELDS.analyst_target_price: None,
                    '_AnalystRatingStrongBuy': None,
                    '_AnalystRatingBuy': None,
                    '_AnalystRatingHold': None,
                    '_AnalystRatingSell': None,
                    '_AnalystRatingStrongSell': None,
                }
            )
        
    @staticmethod
    def _scrape_window_analytics_feed(feed_json: Dict[str, dict], ticker: str, window_size) -> pd.Series:
        if not (
            feed_json["STDDEV"]['payload']['RETURNS_CALCULATIONS']["STDDEV"]["RUNNING_STDDEV"]
            .get(ticker, None)
        ):
            return None
        
        mean_window_var_in_period = np.mean(list(
            feed_json["STDDEV"]['payload']['RETURNS_CALCULATIONS']["STDDEV"]["RUNNING_STDDEV"][ticker].values()
        ))

        # low volatility = hold for window length
        if mean_window_var_in_period < 0.02:
            valid_period = window_size  # trading days

        elif mean_window_var_in_period < 0.05:
            valid_period = window_size // 2

        # high volatility = hold for << window length
        else:
            valid_period = window_size // 4

        return valid_period
    
    @staticmethod
    def _scrape_sentiment_feed(feed_json: Dict[str, dict]) -> pd.Series:
        ticker_occurrences = defaultdict(list)
        for i, article_dict in enumerate(tqdm(feed_json["feed"])):
            for ticker_dict in article_dict["ticker_sentiment"]:
                ticker = ticker_dict["ticker"]
                ticker_occurrences[ticker].append(i)

        occurrences_series = pd.Series(ticker_occurrences, name="_news_occurences", dtype=object)
        return occurrences_series
    
    @staticmethod
    def _collect_sentiment_scores(feed_json: Dict[str, dict], row: pd.Series) -> pd.Series:
        tot_sentiment = []
        tot_relevance = []

        occurrences = row["_news_occurences"]
        if not isinstance(occurrences, list):
            occurrences = []

        for i in occurrences:
            for ticker_dict in feed_json["feed"][i]["ticker_sentiment"]:
                if row.name == ticker_dict["ticker"]:
                    tot_sentiment += [float(ticker_dict["ticker_sentiment_score"])]
                    tot_relevance += [float(ticker_dict["relevance_score"])]

        return pd.Series({
            "_all_sentiment_scores": tot_sentiment,
            "_all_relevance_scores": tot_relevance
        })
    
    @staticmethod
    def _aggregate_scores(row: pd.Series) -> pd.Series:
        df_exploded: pd.DataFrame = (
            row
            .explode(["_all_sentiment_scores", "_all_relevance_scores"])
        )
        df_exploded = (
            df_exploded
            .assign(**{
                "_weighted_sentiment": lambda x: x["_all_sentiment_scores"] * x["_all_relevance_scores"]
            })
            .reset_index()
            .groupby(INPUT_FIELDS.ticker)
            [["_weighted_sentiment", "_all_relevance_scores"]]
            .sum()
            .replace({0: None})
            .dropna()
        )
        return df_exploded["_weighted_sentiment"].divide(df_exploded["_all_relevance_scores"])
    
    @staticmethod
    def _convert_sentiment_score(sentiment_score: float) -> pd.Series:
        
        if sentiment_score > 0.35:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "buy", INPUT_FIELDS.sentiment_conviction: "strong"})
        elif 0.15 < sentiment_score <= 0.35:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "buy", INPUT_FIELDS.sentiment_conviction: "small"})
        elif -0.15 <= sentiment_score <= 0.15:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "hold", INPUT_FIELDS.sentiment_conviction: "strong"})
        elif -0.35 <= sentiment_score < -0.15:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "sell", INPUT_FIELDS.sentiment_conviction: "small"})
        else:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "sell", INPUT_FIELDS.sentiment_conviction: "strong"})
        
    @staticmethod
    def _convert_analyst_score(sentiment_score: float) -> pd.Series:
        
        score = round(sentiment_score, 0)
        if score == 2:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "buy", INPUT_FIELDS.sentiment_conviction: "strong"})
        elif score == 1:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "buy", INPUT_FIELDS.sentiment_conviction: "small"})
        elif score == 0:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "hold", INPUT_FIELDS.sentiment_conviction: "strong"})
        elif score == -1:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "sell", INPUT_FIELDS.sentiment_conviction: "small"})
        elif score == 0:
            return pd.Series({INPUT_FIELDS.sentiment_direction: "sell", INPUT_FIELDS.sentiment_conviction: "strong"})
        else:
            return pd.Series({INPUT_FIELDS.sentiment_direction: None, INPUT_FIELDS.sentiment_conviction: None})