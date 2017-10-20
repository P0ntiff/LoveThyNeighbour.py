#k-nearest neighbours model for language classification

import csv
import re
from string import digits
import numpy
import math

#Encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')


languageLabels = {'0': "Slovak",
				  '1': "French", 
				  '2': "Spanish", 
				  '3': "German", 
				  '4': "Polish"}

#How many examples for each language
exampleCounts = {'0': 14167,
		  '1': 141283,
		  '2': 69974,
		  '3': 37014,
		  '4': 14079}

#Language predicted for empty lines (French as it is the most common)
defaultPrediction = 1

examples = [item[1] for item in list(csv.reader(open("train_set_x.csv", 'rb')))[1:]]
labels = [item[1] for item in list(csv.reader(open("train_set_y.csv", 'rb')))[1:]]


#Returns a list of classifications/predictions for each example in the test set
def main(rawExamples, rawLabels):
	examples, labels = examplePreprocessing(rawExamples, rawLabels)
	tf_idf_scores = buildTF_IDFScores(examples, labels)
	
	#DEBUG
	#for (key, value) in tf_idf_scores.items():
	#	for j in value:
	#		print(key + " - " + str(j))
	#TODO: Come up with distance metric

	return predictions


#Remove digits, extra spaces, emojis, and URLS in the data
#Flatten string into list of characters
#TODO: remove "-", "_", ".", "GIF"
def examplePreprocessing(examples, labels):
	cleanExamples = []
	cleanLabels = []
	#Filter to get rid of all non-multilingual unicode characters (i.e emoji and others)
	characterFilter = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)

	print("...Preprocessing...")
	for i in range(0, len(examples)):
		noEmoji = characterFilter.sub('', unicode(examples[i]))
		flattened = ''.join(noEmoji.split())
		noDigits = str(flattened).translate(None, digits)
		#Check for URLS (check for "http" and if not found check for "www")
		noURLS = ""
		indexOfURL = noDigits.find("http")
		if indexOfURL == -1:
			indexOfURL = noDigits.find("www")
		if indexOfURL != -1:
			noURLS = noDigits[:indexOfURL]
		else:
			noURLS = noDigits

		cleanExamples.append(noURLS)
		cleanLabels.append(labels[i])
		#print(str(cleanLabels[i]) + "- " + str(cleanExamples[i]))
	print("Preprocessing complete.")

	#regularisation
	#scale/normalise dimensions
	#remove noisy/useless examples


	return (cleanExamples, cleanLabels)


#Goes through and builds the tf-idf scores for every character in each language
#Outputs a list of five dictionaries (one for each language) where each list
# contains lists of the 20-highest tf-idf scores for the sentence 
#Precondition: examples is a list of sentences sanitised by examplePreprocessing()
def buildTF_IDFScores(examples, labels):
	print("...Converting training examples to tf-idf vectors....")
	#A count of the total number of examples that contain the character
	documentFrequencies = {"0": {}, "1": {}, "2": {}, "3": {}, "4": {} }
	sentenceCount = len(examples)
	for i in range(0, sentenceCount):
		sentenceLength = len(examples[i])
		foundChars = []
		for j in range(0, sentenceLength):
			#Only want to store the first occurence of a character in a sentence
			if (examples[i][j] in foundChars):
				continue
			else:
				foundChars.append(examples[i][j])
			if (examples[i][j] in documentFrequencies[labels[i]].keys()):
				documentFrequencies[labels[i]][examples[i][j]] += 1
			else:
				documentFrequencies[labels[i]][examples[i][j]] = 1

	listsOfTF_IDFS = {"0": [], "1": [], "2": [], "3": [], "4": []}
	#Get the char's frequency in each sentence and calculate TF-IDF for each char

	for i in range(0, sentenceCount):
		sentenceLength = len(examples[i])
		if (sentenceLength == 0):
			#store something here?
			#should empty lines be used in training at all?
			continue

		#A count of the total frequency of the character within the sentence
		termFrequencies = {}
		for j in range(0, sentenceLength):
			if (examples[i][j] in termFrequencies.keys()):
				termFrequencies[examples[i][j]] += 1
			else:
				termFrequencies[examples[i][j]] = 1

		#calculate the list of tf-idfs for the sentence
		tf_idfs = []
		for j in range(0, sentenceLength):
			#REPLACE DOCUMENT LENGTH WITH LANGUAGE EXAMPLE COUNT
			tf_idf = termFrequencies[examples[i][j]] * math.log(exampleCounts[labels[i]] / documentFrequencies[labels[i]][examples[i][j]])
			tf_idfs.append(tf_idf)

		#(this could be done in vectoriseSentence())
		if (sentenceLength < 20):
			for j in range(0, 20 - sentenceLength):
				#TODO: Think of better approach than appending zeros.
				tf_idfs.append(0)

		#take the 20 largest tf-idfs for the vector
		output_tf_idfs = [0 for i in range(0, 20)]
		currentMinIndex = 0
		for j in range(0, len(tf_idfs)):
			if tf_idfs[j] > output_tf_idfs[currentMinIndex]:
				output_tf_idfs[currentMinIndex] = tf_idfs[j]
				#update min index
				currMin, currMinIndex = min((val, index) for (index, val) in enumerate(output_tf_idfs))
				currentMinIndex = currMinIndex

		#sort in descending order and return
		listsOfTF_IDFS[labels[i]].append(sorted(output_tf_idfs, reverse=True))

	print("Training conversion complete.")
	return listsOfTF_IDFS



def getTF_IDFScore(example, label, index, charIndex):
	return None	





#Get the 20 highest character tf-idf scores (for that language) out of the sentence 
#If <20 characters in the training example then 
# 1) get the tf-idf scores of the n characters there
# 2) fill in the rest with the (20 - n) highest tf-idf scored characters for that language
#Precondition: sentence is a flattened list of chars outputted by examplePreprocessing()
def vectoriseTrainingSentence(sentence):
	return None


def vectoriseTestSentence(sentence):

	#return one vector for each language
	return None



def crossValidationWeighting():
	#repeat by cross-validating and take an average of the predictions
	return None

main(examples, labels)
