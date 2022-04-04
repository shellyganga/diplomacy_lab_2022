#!/usr/bin/python

import sys
import data_pulling
from datetime import datetime

tweet_hist_data = data_pulling.full_query(sys.argv[1], sys.argv[2], sys.argv[3])

print(tweet_hist_data)