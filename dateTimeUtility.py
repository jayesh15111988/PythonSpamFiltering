import os;
import datetime;
import time;
from Constants import DEFAULT_MAXIMUM_DAYS_BEFORE_FILE_DISCARD;

def getMaximumDaysDifferenceBeforeFileGetsOld():
    return DEFAULT_MAXIMUM_DAYS_BEFORE_FILE_DISCARD;

def isTrainingFileOldEnough(filename):
    if(not(os.path.isfile(filename))):
        return True;

    t = os.path.getmtime(filename)
    differenceInSecondsBetweenTwoFiles=(time.time()-t);
    diffrenceInDays=differenceInSecondsBetweenTwoFiles/(24*60*60);
    return diffrenceInDays>getMaximumDaysDifferenceBeforeFileGetsOld();
    #print(t);
    #return datetime.datetime.fromtimestamp(t)
#print(modification_date("productionMessageData"));
#print("Now's date is ",time.time());
print("Is training file old enough to discard?",isTrainingFileOldEnough("productionMessageData"));

