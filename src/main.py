from src.signal_generation.information_collation_task import InformationCollationPipeline


if __name__ == "__main__":
    
    _obj = InformationCollationPipeline(api_key=None, tickers=["TSLA", "IBM"])
    _obj.run()