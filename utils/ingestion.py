import datetime
import uuid


def get_ingestion_id() -> str:
    """Generate a unique ingestion ID based on UUID and current timestamp.

    Returns:
        str: A unique ingestion ID.
    """
    return f"{uuid.uuid4()}_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d%H%M')}"