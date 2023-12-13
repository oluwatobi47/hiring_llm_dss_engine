import datetime
import random


def generate_random_date(start_date, end_date):
    """ Generates a random date within the specified start and end_date """

    # Calculate the date difference and generate a random number of days within the date range
    date_diff = end_date - start_date
    random_days = datetime.timedelta(days=random.randrange(date_diff.days))

    # Generate the random date
    return datetime(start_date + random_days)