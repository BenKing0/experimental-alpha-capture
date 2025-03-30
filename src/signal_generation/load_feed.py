from typing import List, Dict, Optional
from collections import defaultdict

import requests
import pickle
import os


class FeedLoader:

    def __init__(
        self,
        api_key: Optional[str] = None,
        pickle_dir: Optional[str] = None,
    ):
        self.api_key = api_key
        self.pickle_dir = pickle_dir

        if not self.api_key:
            if self.pickle_dir:
                self.load_locally = True
            else:
                raise ReferenceError("Need to either pass API key or pickle location")
            
        else:
            self.load_locally = False

            if self.pickle_dir:
                self.save_locally = True
            else:
                self.save_locally = False

    def load_news_sentiment(self, tickers: List[str]) -> Dict[str, dict]:
        
        if not self.load_locally:
            feed = APIs.get_news_sentiment(self.api_key, tickers=tickers)
            
            if self.save_locally:
                with open(
                    os.path.join(self.pickle_dir, "news_sentiment.pkl"), "wb"
                ) as f:
                    pickle.dump(feed, f)

            return feed
        
        with open(
            os.path.join(self.pickle_dir, "news_sentiment.pkl"), "rb"
        ) as f:
            feed = pickle.load(f)

        return feed
    
    def load_fundamentals(self, tickers: List[str]) -> Dict[str, dict]:
        
        if not self.load_locally:
            feed = APIs.get_fundamentals(self.api_key, tickers=tickers)
            
            if self.save_locally:
                with open(
                    os.path.join(self.pickle_dir, "fundamentals.pkl"), "wb"
                ) as f:
                    pickle.dump(feed, f)

            return feed
        
        with open(
            os.path.join(self.pickle_dir, "fundamentals.pkl"), "rb"
        ) as f:
            feed = pickle.load(f)

        return feed
    
    def load_insider_moves(self, tickers: List[str]) -> Dict[str, dict]:
        
        if not self.load_locally:
            feed = APIs.get_insider_moves(self.api_key, tickers=tickers)
            
            if self.save_locally:
                with open(
                    os.path.join(self.pickle_dir, "insider_moves.pkl"), "wb"
                ) as f:
                    pickle.dump(feed, f)

            return feed
        
        with open(
            os.path.join(self.pickle_dir, "insider_moves.pkl"), "rb"
        ) as f:
            feed = pickle.load(f)

        return feed
    
    def load_indicator(
        self, 
        tickers: List[str],
        indicator: str = "RSI", 
        interval: str = "daily", 
        time_period: int = 14,
    ) -> Dict[str, dict]:
        
        if not self.load_locally:
            feed = APIs.get_technical_indicator(
                self.api_key, 
                tickers=tickers,
                indicator=indicator,
                interval=interval,
                time_period=time_period,
            )
            
            if self.save_locally:
                with open(
                    os.path.join(self.pickle_dir, f"{indicator.lower()}_feed.pkl"), "wb"
                ) as f:
                    pickle.dump(feed, f)

            return feed
        
        with open(
            os.path.join(self.pickle_dir, f"{indicator.lower()}_feed.pkl"), "rb"
        ) as f:
            feed = pickle.load(f)

        return feed
    
    def load_window_analytics(
        self, 
        tickers: List[str],
        n_month: int,
        calculations: List[str],
        window: int = 20,
        interval: str = "DAILY",
    ) -> Dict[str, dict]:
        
        if not self.load_locally:
            feed = APIs.get_window_analytics(
                self.api_key, 
                tickers=tickers,
                n_month=n_month,
                calculations=calculations,
                window=window,
                interval=interval,
            )
            
            if self.save_locally:
                with open(
                    os.path.join(self.pickle_dir, "sliding_window_analytics.pkl"), "wb"
                ) as f:
                    pickle.dump(feed, f)

            return feed
        
        with open(
            os.path.join(self.pickle_dir, "sliding_window_analytics.pkl"), "rb"
        ) as f:
            feed = pickle.load(f)

        return feed


class APIs:

    base_url = "https://www.alphavantage.co/query"

    @staticmethod
    def _get_comma_seperated(_list: List[str]):
        """
        Helper function for turning list of strings into comma seperated single strings for API call
        """
        if len(_list) == 0:
            _list = _list[0]
        elif len(_list) > 0:
            _list = ",".join(_list)
        else:
            raise ReferenceError(f"Need to specify at least one item. Received {_list}.")
        return _list

    @staticmethod
    def get_news_sentiment(api: str, tickers: List[str]) -> Dict[str, dict]:
        """
        Returns a data JSON of the format:

        {
        "items": "50",
        "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x <= 0.15: Neutral; 0.15 < x <= 0.35: Somewhat-Bullish; x > 0.35: Bullish",
        "relevance_score_definition": "0 < x <= 1, with a higher score indicating higher relevance.",
        "feed": [
            {
            "title": "Sample News Title",
            "url": "https://www.example.com/news-article",
            "time_published": "20250328T200000",
            "authors": ["Author Name"],
            "summary": "Brief summary of the news article.",
            "banner_image": "https://www.example.com/image.jpg",
            "source": "News Source",
            "category_within_source": "Finance",
            "source_domain": "example.com",
            "topics": [
                {
                "topic": "Financial Markets",
                "relevance_score": "0.9"
                }
            ],
            "overall_sentiment_score": "0.25",
            "overall_sentiment_label": "Somewhat-Bullish",
            "ticker_sentiment": [
                {
                "ticker": "AAPL",
                "relevance_score": "0.8",
                "ticker_sentiment_score": "0.3",
                "ticker_sentiment_label": "Somewhat-Bullish"
                }
            ]
            }
            // Additional articles...
        ]
        }
        """
        tickers = APIs._get_comma_seperated(tickers)
        params = {
            "apikey": api,
            "symbol": tickers,
            "function": "NEWS_SENTIMENT"
        }
        return requests.get(APIs.base_url, params=params).json()

    @staticmethod
    def get_insider_moves(api: str, tickers: List[str]) -> Dict[str, dict]:
        """
        Returns a data JSON of the format:

        {
        "symbol": "AAPL",
        "insider_trades": [
            {
            "name": "John Doe",
            "relationship": "CEO",
            "transaction_date": "2025-03-27",
            "transaction_code": "P",
            "transaction_amount": "1000",
            "transaction_price": "150.00",
            "transaction_share": "100",
            "post_transaction_share": "10100"
            }
            // Additional transactions...
        ]
        }

        """
        tickers = APIs._get_comma_seperated(tickers)
        params = {
            "apikey": api,
            "symbol": tickers,
            "function": "INSIDER_TRANSACTIONS"
        }
        return requests.get(APIs.base_url, params=params).json()

    @staticmethod
    def get_window_analytics(
        api: str, 
        tickers: List[str],
        n_month: int,
        calculations: List[str],
        window: int,
        interval: str,
    ) -> Dict[str, dict]:
        """
        Returns a data JSON of the form:

        {
        "Meta Data": {
            "1. Information": "Intraday (5min) open, high, low, close prices and volume",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2025-03-28 20:00:00",
            "4. Interval": "5min",
            "5. Output Size": "Compact",
            "6. Time Zone": "US/Eastern"
        },
        "Time Series (5min)": {
            "2025-03-28 20:00:00": {
            "1. open": "150.00",
            "2. high": "151.00",
            "3. low": "149.50",
            "4. close": "150.50",
            "5. volume": "1000"
            }
            // Additional time intervals...
        }
        }

        """
        tickers = APIs._get_comma_seperated(tickers)
        calculations = APIs._get_comma_seperated(calculations)
        url = f"""
            https://alphavantageapi.co/timeseries/running_analytics
        """
        params = {
            "SYMBOLS": tickers,
            "RANGE": f"{n_month}month",
            "INTERVAL": interval,
            "OHLC": "close",
            "WINDOW_SIZE": window,
            "CALCULATIONS": calculations,
            "apikey": api,
            "function": "ANALYTICS_SLIDING_WINDOW",
        }
        return requests.get(APIs.base_url, params=params).json()
    
    @staticmethod
    def get_technical_indicator(
            api: str,
            tickers: List[str],
            indicator: str, 
            interval: str, 
            time_period: int,
        ) -> Dict[str, dict]:
        """
        Returns a JSON of the form:

        {
            "Meta Data": {
                "1: Symbol": "IBM",
                "2: Indicator": "Relative Strength Index (RSI)",
                "3: Last Refreshed": "2025-03-28 20:00:00",
                "4: Interval": "daily",
                "5: Time Period": 14,
                "6: Series Type": "close",
                "7: Time Zone": "US/Eastern Time"
            },
            "Technical Analysis: RSI": {
                "2025-03-28": {
                    "RSI": "56.1234"
                },
                "2025-03-27": {
                    "RSI": "54.9876"
                }
                // Additional data points...
            }
        }
        """
        output = defaultdict(dict)
        for ticker in tickers:
            params = {
                "function": indicator,
                "symbol": ticker,
                "interval": interval,
                "time_period": time_period,
                "series_type": "close",
                "apikey": api,
            }
            output[ticker] = requests.get(APIs.base_url, params=params).json()
        return output

    @staticmethod
    def get_fundamentals(api: str, tickers: List[str]) -> Dict[str, dict]:
        """
        Returns a JSON of the form:

        {
            "Symbol": "IBM",
            "AssetType": "Common Stock",
            "Name": "International Business Machines Corporation",
            "Description": "IBM is a global technology company...",
            "CIK": "0000051143",
            "Exchange": "NYSE",
            "Currency": "USD",
            "Country": "USA",
            "Sector": "Technology",
            "Industry": "Information Technology Services",
            "Address": "1 New Orchard Road, Armonk, NY, USA",
            "FiscalYearEnd": "December",
            "LatestQuarter": "2025-03-31",
            "MarketCapitalization": "120000000000",
            "EBITDA": "15000000000",
            "PERatio": "14.50",
            "PEGRatio": "1.20",
            "BookValue": "20.50",
            "DividendPerShare": "6.00",
            "DividendYield": "0.045",
            "EPS": "10.00",
            "RevenuePerShareTTM": "50.00",
            "ProfitMargin": "0.15",
            "OperatingMarginTTM": "0.18",
            "ReturnOnAssetsTTM": "0.05",
            "ReturnOnEquityTTM": "0.20",
            "RevenueTTM": "75000000000",
            "GrossProfitTTM": "35000000000",
            "DilutedEPSTTM": "9.50",
            "QuarterlyEarningsGrowthYOY": "0.05",
            "QuarterlyRevenueGrowthYOY": "0.03",
            "AnalystTargetPrice": "140.00",
            "TrailingPE": "14.00",
            "ForwardPE": "13.00",
            "PriceToSalesRatioTTM": "1.60",
            "PriceToBookRatio": "6.80",
            "EVToRevenue": "1.80",
            "EVToEBITDA": "10.00",
            "Beta": "1.10",
            "52WeekHigh": "150.00",
            "52WeekLow": "100.00",
            "50DayMovingAverage": "120.00",
            "200DayMovingAverage": "125.00",
            "SharesOutstanding": "1000000000",
            "DividendDate": "2025-04-15",
            "ExDividendDate": "2025-03-30"
        }
        """
        output = defaultdict(dict)

        for ticker in tickers:
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": api,
            }
            ticker_feed = requests.get(APIs.base_url, params=params).json()
            output[ticker] = ticker_feed
        
        return output