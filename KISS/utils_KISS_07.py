from pandas import Timestamp
from typing import Optional

def get_timestamp(date_only: Optional[bool] = False, time_only: Optional[bool] = False, for_seed: Optional[bool] = False) -> str:
    """Return the current timestamp in ISO 8601 format.

    Args:
        date_only (bool): If True, return only the date in YYYY-MM-DD format.
        time_only (bool): If True, return only the time in HHMMSS format.
        for_seed (bool): If True, return the timestamp as an integer for use as a seed value in the format HHMMSS.

    Returns:
        str: The current timestamp.
    """
    if date_only:
        return Timestamp.now(tz=None).strftime('%Y-%m-%d')
    elif time_only:
        return Timestamp.now(tz=None).strftime('%H%M%S')
    elif for_seed:
        return int(Timestamp.now(tz=None).strftime('%H%M%S'))
    else:
        return Timestamp.now(tz=None).isoformat(timespec='seconds')
