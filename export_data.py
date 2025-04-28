import json
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
import os
from data_config import MONGO_URI, MONGO_DB_NAME, PRICE_COLLECTION, VOLUME_COLLECTION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_data(collection, companies, output_file, n_days=200):
    """Export MongoDB data to JSON."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        coll = db[collection]
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=n_days)
        query = {
            "company": {"$in": companies},
            "date": {"$gte": start_date, "$lte": end_date}
        }
        data = list(coll.find(query).sort("date", 1))
        client.close()
        with open(output_file, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Exported {len(data)} records to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise

def main():
    """Export data."""
    export_data(PRICE_COLLECTION, ["AAPL", "MSFT", "NVDA", "AMD"], os.path.dirname(__file__) + "/data/prices.json", 300)
    export_data(VOLUME_COLLECTION, ["AAPL", "MSFT", "NVDA", "AMD"], os.path.dirname(__file__) + "/data/volumes.json", 300)

if __name__ == "__main__":
    main()