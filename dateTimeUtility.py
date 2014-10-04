import os;
import datetime;
import time;
from Constants import DEFAULT_MAXIMUM_DAYS_BEFORE_FILE_DISCARD;

def getMaximumDaysDifferenceBeforeFileGetsOld():
    return DEFAULT_MAXIMUM_DAYS_BEFORE_FILE_DISCARD;

def isTrainingFileOldEnough(filename):
    if(not(os.path.isfile(filename))):
        return True;

    lastTimeFileModified = os.path.getmtime(filename)
    differenceInSecondsBetweenTwoFiles=(time.time()-lastTimeFileModified);
    #Converting Seconds to days. 1 Day - 24*60*60 seconds
    diffrenceInDays=differenceInSecondsBetweenTwoFiles/(24*60*60);
    print("Last time File containing training data modified was ",diffrenceInDays," Days ago");
    return diffrenceInDays>getMaximumDaysDifferenceBeforeFileGetsOld();


