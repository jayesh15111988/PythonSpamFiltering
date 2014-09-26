import Constants;
from StringUtilities import *;
import re;

def setProbabilityForOccurrenceOfEachWordInStore(inputStoreWithCountOfEachWord):
    totalFrequencyOfAllWordsInStore=0;
    collectionOfAllWordsInStore=inputStoreWithCountOfEachWord.keys();

    for individualKey in collectionOfAllWordsInStore:
        totalFrequencyOfAllWordsInStore=totalFrequencyOfAllWordsInStore+inputStoreWithCountOfEachWord[individualKey];

    for word in collectionOfAllWordsInStore:
        inputStoreWithCountOfEachWord[word]=inputStoreWithCountOfEachWord[word]/totalFrequencyOfAllWordsInStore;
       
def isMessageSpam(receivedMessage,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,probabilityOfRegularMessage,probabilityOfSpamMessage):
    tokenizedInputString = re.split('\W+',getStringWithPunctuationsRemoved(receivedMessage));

    #Initialize probability values first
    temporaryProbabilityThatMessageIsSpam=Constants.DEFAULT_PROBABILITY_VALUE_FOR_ITEMS;
    temporaryProbabilityThatMessageIsNotSpam=Constants.DEFAULT_PROBABILITY_VALUE_FOR_ITEMS;

    #Now using Actual Bayes Theorm;

    #This is shady - They say use zero or any marginal value using minimum value among them all and suppress it by 1/10th of its original value
    lowestProbabilityValue=(min(min(frequencyOfWordsInRegularMessages.values()),min(frequencyOfWordsInSpamMessages.values())))/10;
        
    for individualTokenInInputMessage in tokenizedInputString:

        if not(individualTokenInInputMessage in frequencyOfWordsInRegularMessages):
            temporaryProbabilityThatMessageIsNotSpam=temporaryProbabilityThatMessageIsNotSpam*lowestProbabilityValue;
        else:
            temporaryProbabilityThatMessageIsNotSpam=temporaryProbabilityThatMessageIsNotSpam*frequencyOfWordsInRegularMessages[individualTokenInInputMessage];
            
        if not(individualTokenInInputMessage in frequencyOfWordsInSpamMessages):
            temporaryProbabilityThatMessageIsSpam=temporaryProbabilityThatMessageIsSpam*lowestProbabilityValue;
        else:
            temporaryProbabilityThatMessageIsSpam=temporaryProbabilityThatMessageIsSpam*frequencyOfWordsInSpamMessages[individualTokenInInputMessage];     

    finalProbabilityThatMessageIsNotSpam=temporaryProbabilityThatMessageIsNotSpam*probabilityOfRegularMessage;
    finalProbabilityThatMessageIsSpam=temporaryProbabilityThatMessageIsSpam*probabilityOfSpamMessage;
    print("P(Message)--> ",finalProbabilityThatMessageIsNotSpam," And P(spam) --> ",finalProbabilityThatMessageIsSpam,"\n\n");
    return finalProbabilityThatMessageIsSpam > finalProbabilityThatMessageIsNotSpam;
