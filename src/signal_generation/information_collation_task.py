from typing import List, Dict
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from src.schemas import FIELDS
from src.signal_generation.load_feed import FeedLoader

PICKLE_DIRECTORY = "notebooks/pickled_feeds"


class InformationCollationPipeline:
    
    def __init__(self, api_key: str, tickers: List[str]) -> None:
        self.api_key = api_key
        self.tickers = tickers

        self.signals_df = pd.DataFrame(columns=[
            FIELDS.ticker,
            FIELDS.sentiment_score,
            FIELDS.direction,
            FIELDS.size,
            FIELDS.conviction,
            FIELDS.execution_time,
            FIELDS.holding_period,
        ])
        self.signals_df[FIELDS.ticker] = pd.Series(tickers)
        self.signals_df.set_index(FIELDS.ticker, inplace=True)

        self.loader = FeedLoader(api_key, pickle_dir=PICKLE_DIRECTORY)

    def run(self) -> pd.DataFrame:
        ...

    def process_analytics(self) -> None:
        feed_json = self.loader.load_window_analytics(
            self.tickers,
            n_month=12,
            calculations=[
                "MEAN",
                "MEDIAN",
                "STDDEV",
                "CORRELATION",
            ],
            window=20,
            interval="DAILY"
        )

        ...

    def process_fundamentals(self) -> None:
        feed_json = self.loader.load_fundamentals(self.tickers)

        ...

    def process_sentiment(self) -> None:
        feed_json = self.loader.load_news_sentiment(self.tickers)
        
        self.signals_df["_news_occurences"] = self._scrape_feed(feed_json)
        self.signals_df[["_all_sentiment_scores", "_all_relevance_scores"]] = self.signals_df.apply(
            lambda row: self._collect_sentiment_scores(feed_json, row),
            axis=1,
        )

        self.signals_df[FIELDS.sentiment_score] = self._aggregate_scores(self.signals_df)
        self.signals_df[[FIELDS.direction, FIELDS.conviction]] = (
            self.signals_df[FIELDS.sentiment_score]
            .dropna()
            .apply(self._convert_score)
        )

        self.signals_df.drop(columns=[
            "_news_occurences",
            "_all_sentiment_scores",
            "_all_relevance_scores",
        ], inplace=True)
    
    @staticmethod
    def _scrape_feed(feed_json: Dict[str, dict]) -> pd.Series:
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
            .groupby(FIELDS.ticker)
            [["_weighted_sentiment", "_all_relevance_scores"]]
            .sum()
            .replace({0: None})
            .dropna()
        )
        return df_exploded["_weighted_sentiment"].divide(df_exploded["_all_relevance_scores"])
    
    @staticmethod
    def _convert_score(sentiment_score: float) -> pd.Series:
        
        if sentiment_score > 0.35:
            return pd.Series({FIELDS.direction: "buy", FIELDS.conviction: "very strong"})
        elif 0.15 < sentiment_score <= 0.35:
            return pd.Series({FIELDS.direction: "buy", FIELDS.conviction: "good"})
        elif -0.15 <= sentiment_score <= 0.15:
            return pd.Series({FIELDS.direction: "hold", FIELDS.conviction: "small"})
        elif -0.35 <= sentiment_score < -0.15:
            return pd.Series({FIELDS.direction: "sell", FIELDS.conviction: "good"})
        else:
            return pd.Series({FIELDS.direction: "sell", FIELDS.conviction: "very strong"})