import logging

from ingestion.ingestion_manager import IngestionManager
from ingestion.ingestors.blog_ingestor import BlogIngestor
from ingestion.ingestors.cook_manual_ingestor import CookManualIngestor
from ingestion.ingestors.distances_ingestor import DistancesIngestor
from ingestion.ingestors.galactic_code_ingestor import GalacticCodeIngestor
from ingestion.ingestors.menu_ingestor import MenuIngestor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def start_ingestion():
    # Initialise all databases and Qdrant collections, and prepare the ingestion context
    with IngestionManager.create() as ingestion_manager:
        log.info("✅ All databases and Qdrant collections initialised successfully.")
        # Start ingestion
        DistancesIngestor().ingest_all(ingestion_manager)
        log.info("✅ Distances Ingestion completed successfully.")
        MenuIngestor().ingest_all(ingestion_manager)
        log.info("✅ Menu Ingestion completed successfully.")
        GalacticCodeIngestor().ingest_all(ingestion_manager)
        log.info("✅ Galactic Code Ingestion completed successfully.")
        CookManualIngestor().ingest_all(ingestion_manager)
        log.info("✅ Cook Manual Ingestion completed successfully.")
        BlogIngestor().ingest_all(ingestion_manager)
        log.info("✅ Blog Ingestion completed successfully.")


if __name__ == "__main__":
    start_ingestion()
