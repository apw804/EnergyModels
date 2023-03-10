from pandas import Timestamp

def get_timestamp() -> str:
    """Return the current timestamp in ISO 8601 format."""
    return Timestamp.now(tz=None).isoformat(timespec='seconds')
