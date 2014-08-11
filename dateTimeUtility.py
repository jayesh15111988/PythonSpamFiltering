import os
import datetime;
import time;
def getMaximumDaysDifferenceBeforeFileGetsOld():
    return 4;

def isTrainingFileOldEnough(filename):
    t = os.path.getmtime(filename)
    differenceInSecondsBetweenTwoFiles=(time.time()-t);
    diffrenceInDays=differenceInSecondsBetweenTwoFiles/(24*60*60);
    return diffrenceInDays>getMaximumDaysDifferenceBeforeFileGetsOld();
    #print(t);
    #return datetime.datetime.fromtimestamp(t)
#print(modification_date("productionMessageData"));
#print("Now's date is ",time.time());
print(isTrainingFileOldEnough("productionMessageData"));

