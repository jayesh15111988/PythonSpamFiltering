import random;
import math;

def GenerateKMeansClusters(collectionOfVectorsOfAllMessages,numberOfDesiredOutputClusters):
    #Beginning of computation for K-means clustering for given set of message - Gotta be challenging
    numberOfClusters=2 #Since there are only two classes for us

    
    #print("Input Sequence is ",collectionOfVectorsOfAllMessages);
    
    centroidHolderForInputMessages=[];
    numberOfClustersGeneratedSoFar=0;
    duplicateChecker={};
    
    maximumVectorSequenceValue=len(collectionOfVectorsOfAllMessages)-1
    #print("Maximum length of collection vector is Modified ** ",maximumVectorSequenceValue);

    while(numberOfClustersGeneratedSoFar<numberOfDesiredOutputClusters):
        #generate random point first range is
        
        randomVectorSequence=random.randint(0, maximumVectorSequenceValue);
        
        if(randomVectorSequence in duplicateChecker):
            continue;
        else:
            centroidHolderForInputMessages.append(collectionOfVectorsOfAllMessages[randomVectorSequence]);
            numberOfClustersGeneratedSoFar=numberOfClustersGeneratedSoFar+1;
            duplicateChecker[randomVectorSequence]=1;
    #Both centroids thus generated, append -1 to them until for both of them length is equal to
    #maximum length of vector in given collection
    lengthOfMaximumLengthVectorInCollection=len(max(collectionOfVectorsOfAllMessages, key=len));  #len(collectionOfVectorsOfAllMessages[maximumVectorSequenceValue]);
    print("max length is ",lengthOfMaximumLengthVectorInCollection);
    newCentroidHolderForInputMessages=[];
    
    #print("max length is  ",lengthOfMaximumLengthVectorInCollection);
    for individualCentroids in centroidHolderForInputMessages:
        while(len(individualCentroids)<lengthOfMaximumLengthVectorInCollection):
            individualCentroids.append(-1);
    #We made sure, it won't cause index out of bound for an incomplete centroid vectors
    #In case of Other vectors, we will simple check against index and their length.
    # If desired index >(=)len(vector)-1(len(vector)) then don't access elemnt simply return -1 as default value
    #print("updated max length is",len(centroidHolderForInputMessages[0]),len(centroidHolderForInputMessages[1]));
    #print("Generated Centroid Sequence is ",centroidHolderForInputMessages," and record sequence number is ",duplicateChecker);
    #print("Original centroid is ",centroidHolderForInputMessages);
    #Make sure we run this loop until entire sum of difference between previous and current centroid is less
    #than some predefined threshold  - In other words, all of them have been stabilized for sure to specific set of centroid
    predefinedThreshold=5.0;
    differenceBetweenCentroids=1000;

    

    
    

    #TODO - Use NumPy which uses low level C API for Mathematical computations

    
    index=2;
    
    #while(differenceBetweenCentroids>predefinedThreshold):
    iterator1=8;
    #while(iterator1>0):
    while(differenceBetweenCentroids>1.0):
        regularMessagesVectorContainer=[];
        spamMessagesVectorContainer=[];
        allMessagesContainer=[regularMessagesVectorContainer,spamMessagesVectorContainer];    
        #print ("Data BEFORE soritng out centroid is -->  ",'[%s]' % ','.join(map(str, allMessagesContainer)));
        print("Centroid for our purpose is CCCCCCCCCCCCCCCCCCCCCCC ",centroidHolderForInputMessages);
        for individualVector in collectionOfVectorsOfAllMessages:
            minDistanceFromCentroid=1000;
            centroidNumberToChoose=-1;
            euclideanDistanceFromCentroidForGivenVector=0;
            for outerIndex,individualCentroidHolderForInputMessages in enumerate(centroidHolderForInputMessages):
                for index,individualFeatureValueInFeatureVector in enumerate(individualCentroidHolderForInputMessages):
                    
                    individualVectorAttributeValue=individualVector[index] if index < len(individualVector) else -1
                    individualCentroidPointValue=individualCentroidHolderForInputMessages[index];
                    
                        
                    euclideanDistanceFromCentroidForGivenVector=euclideanDistanceFromCentroidForGivenVector+(individualCentroidPointValue- individualVectorAttributeValue)**2       
                euclideanDistanceFromCentroidForGivenVector=(euclideanDistanceFromCentroidForGivenVector)**(0.5);
                if(euclideanDistanceFromCentroidForGivenVector<minDistanceFromCentroid):
                    minDistanceFromCentroid=euclideanDistanceFromCentroidForGivenVector;
                    centroidNumberToChoose=outerIndex;
            if(centroidNumberToChoose!=-1):
                allMessagesContainer[centroidNumberToChoose].append(individualVector);
        #Calculate average of all points in respective centroid holders and reassign them as centroids
        index=index-1;
    #print("All centroids data is "+(', '.join(allMessagesContainer[0])));
        print ("Data after soritng out centroid is -->  ",'[%s]' % ', '.join(map(str, allMessagesContainer)));
    #Previous centroid that we had - Now calculate new centroid from the array we have and compare new centroid with older ones
        lengthOfVector=0;
        newCentroidHolderForPoints=[];
        lengthOfVector=lengthOfMaximumLengthVectorInCollection;
        for index,individualCalculatedVector in enumerate(allMessagesContainer):
            tempHolderArrayForCentroids=[];
            #len(individualCalculatedVector[0]);
            lengthOfVectorInVector=len(individualCalculatedVector);
            #print("Length is ",lengthOfVector);
            vectorWithAverageValueInit=[];
            for indexLevelOne in range(0,lengthOfVector):
                totalValue=0
                for indexLevelTwo,individualPointVector in enumerate(individualCalculatedVector):
                #for individualIndexInsideVector in
                    valToAdd=individualPointVector[indexLevelOne] if indexLevelOne < len(individualPointVector) else -1;
                    totalValue=totalValue+valToAdd;
                vectorWithAverageValueInit.append(totalValue/lengthOfVectorInVector);
            newCentroidHolderForPoints.append(vectorWithAverageValueInit);
        print("old centroid holder with points is ",centroidHolderForInputMessages,len(centroidHolderForInputMessages),len(centroidHolderForInputMessages[0]),len(centroidHolderForInputMessages[1]));
        print("New centroid holder with points is ",newCentroidHolderForPoints,len(newCentroidHolderForPoints),len(newCentroidHolderForPoints[0]),len(newCentroidHolderForPoints[1]));
        #Now calculate difference between old and new centroids

        #Now Append -1 to empty centroid value
        
        
        sumOfDifferencesForCentroids=0.0;
        for indexOuttermost in range(0,numberOfDesiredOutputClusters):
            oldIndividualCentroid=centroidHolderForInputMessages[indexOuttermost];
            newIndividualCentroid=newCentroidHolderForPoints[indexOuttermost];
            for indexInIndividualVector in range(0,lengthOfVector):
                #print("Sum of differecnes is",sumOfDifferencesForCentroids);
                sumOfDifferencesForCentroids=sumOfDifferencesForCentroids+math.fabs(oldIndividualCentroid[indexInIndividualVector]-newIndividualCentroid[indexInIndividualVector]);        
            #sumOfDifferencesForCentroids=sumOfDifferencesForCentroids+fabs(centroidHolderForInputMessages[indexOuttermost]-newCentroidHolderForPoints[indexOuttermost]);
        differenceBetweenCentroids=sumOfDifferencesForCentroids;
        print("Difference at the bottom is ",differenceBetweenCentroids);
        
        centroidHolderForInputMessages=newCentroidHolderForPoints;
        #iterator1=iterator1-1;


    
#GenerateKMeansClusters([[1,2,3,4,3,6,8,9,0],[6,5,4,3,2,1],[4,5,6,3,2,1],[6,5,4,1,2,3],[100,200,300,456,432,291],[300,400,200,100,500,300]],2);
'''
        newCentroidHolderForPoints=[];
        for index,individualCalculatedVector in enumerate(allMessagesContainer):
            tempHolderArrayForCentroids=[];
            for innerIndex,individualCalculatedVectorPoint in enumerate(individualCalculatedVector):
                totalValueOfFeatureAttribute=0;
                #print("value in doubt ",individualCalculatedVectorPoint);
                lengthOfDataPoints=len(individualCalculatedVectorPoint);
                for featureNumber in range(0,lengthOfDataPoints):
                    print("index is ",featureNumber," ",individualCalculatedVectorPoint[featureNumber]);
                    totalValueOfFeatureAttribute=totalValueOfFeatureAttribute+individualCalculatedVectorPoint[featureNumber];
                print("total value of feature is ",totalValueOfFeatureAttribute);
                averageValueOfNthFeature=totalValueOfFeatureAttribute/lengthOfDataPoints;
                tempHolderArrayForCentroids.append(averageValueOfNthFeature);
            newCentroidHolderForPoints.append(tempHolderArrayForCentroids);
        print ("Data after soritng out centroid is -->  ",'[%s]' % ', '.join(map(str, newCentroidHolderForPoints)));
'''








                
                
            
            
