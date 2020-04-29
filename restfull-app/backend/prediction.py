from datetime import datetime

def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))

# Data to serve with our API
RECORDS = [
	{
	    "timestamp": get_timestamp()
	    "model": "1",
	    "serial_number": "s1",
            "capacity_bytes": 100000000000,
            "failure": 0,
            "prediction": 0,
	},
    ]

def read():
    """
    This function responds to a request for /api/people
    with the complete lists of people

    :return:        sorted list of people
    """
    # Create the list of people from our data
    return RECORDS
