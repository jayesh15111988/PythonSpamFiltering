import os
import datetime;
import time;
def modification_date(filename):
    t = os.path.getmtime(filename)
    return t;
    #print(t);
    #return datetime.datetime.fromtimestamp(t)
#print(modification_date("productionMessageData"));
#print("Now's date is ",time.time());
print(time.time()>modification_date("productionMessageData"));
