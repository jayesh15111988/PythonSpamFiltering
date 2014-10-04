#Collection of constants used in the program

DEFAULT_PROBABILITY_VALUE_FOR_ITEMS = 1.0;
OUTPUT_CENTROID_FILE_NAME = "../centroidHolder.txt";
OUTPUT_VECTOR_FILE_NAME = "../categorizedPointsHolder.txt";
OUTPUT_MESSAGE_CATEGORIZATION_FILE_NAME = "messageCategorization.txt";
DEFAULT_CENTROID_TO_ASSIGN_TO_DATA_POINT = -1;
DEFAULT_MINIMUM_DISTANCE_FROM_CENTROID=1000;
DEFAULT_MAXIMUM_DAYS_BEFORE_FILE_DISCARD=60;
TRAINING_DATA_FILE = 'SMSSpamCollection';
SMALL_TRAINING_DATA_FILE = 'sample';
PRODUCTION_DATA_FILE = 'productionMessageData';
DEFAULT_MESSAGE_CHARACTER = 'm';
DEFAULT_SPAM_CHARACTER = 's';

#User might need to customize this based on the label values input training file uses
#For spam and regular messages

DEFAULT_SPAM_INDICATOR_STRING = 'spam';
DEFAULT_MESSAGE_INDICATOR_STRING = 'ham';
OUTPUT_REGULAR_MESSAGES_WORD_FREQUENCY="../regularMessagesWordFrequency.txt"
OUTPUT_SPAM_MESSAGES_WORD_FREQUENCY="../spamMessagesWordFrequency.txt"
