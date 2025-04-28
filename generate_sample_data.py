from pymongo import MongoClient
from datetime import datetime, timedelta
import numpy as np

from data_config import MONGO_URI, MONGO_DB_NAME, PRICE_COLLECTION, VOLUME_COLLECTION
# MongoDB configuration

def generate_sample_data(company="Apple", days=200, loc = 100, scale = 10):
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    price_coll = db[PRICE_COLLECTION]
    volume_coll = db[VOLUME_COLLECTION]

    # Clear existing data
    price_coll.delete_many({"company": company})
    volume_coll.delete_many({"company": company})

    # Generate synthetic data
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    prices = np.random.normal(loc, scale, days).cumsum() / 100 + 100
    volumes = np.random.randint(100000, 1000000, days)

    # generating fake stock prices and volumes
    price_docs = [
        {
            "company": company,
            "stock_price": float(prices[i]),
            "date": base_date - timedelta(days=days - i - 1)
        }
        for i in range(days)
    ]
    volume_docs = [
        {
            "company": company,
            "volume": int(volumes[i]),
            "date": base_date - timedelta(days=days - i - 1)
        }
        for i in range(days)
    ]

    price_coll.insert_many(price_docs)
    volume_coll.insert_many(volume_docs)
    client.close()
    print(f"Inserted {days} days of sample data for {company}")

if __name__ == "__main__":
    generate_sample_data("AAPL", 300, loc=100, scale=10)
    generate_sample_data("MSFT", 300, loc=150, scale=5)
    generate_sample_data("AMD", 300, loc=120, scale=15)
    generate_sample_data("AMZN", 300, loc=95, scale=20)
    generate_sample_data("NVDA", 300, loc=130, scale=12)