import sys
def main(argv):
	import nltk
	import random, re
	import twokenize_wrapper as tok
	import pickle


	stopWords = {}
	st = open('stopWordsNew.txt', 'r')
	inputFile = open(argv[0],'r')
	outputFile = open(argv[1],'w')
	maxEntObjectFile = open('maxEntObject.pkl','rb')

	for line in st:
		line = line.strip('\n')
		if(not stopWords.has_key(line)):
			stopWords[line] = 1

	def featureFunc(tweet):
		feat = {}
		for word in tweet:
			feat[word] = 1
		return feat

	wnl = nltk.stem.WordNetLemmatizer()
	wordListMap = {}
	tokenizedTweets = []
	totalTweets = 0

	for line in inputFile:
		tweet = line.strip('\n')
		tweet = tweet.lower()
		tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 				#HASH TAG
		tweet = re.sub(r'(@[\w]+)','_HANDLE_',tweet)			#HANDLE
		tweet = re.sub(r'http[\w://.~?=%&-]+','_URL_',tweet)	#URL
		tweet = re.sub(r'(/|:|&|\(|\))',' ',tweet) 				# / : & ( ) spaced
		tweet = re.sub(r'(\d+)',r' \1 ',tweet) 					# Digit clusters spaced
		tweet = re.sub(r'(\w+)(-|;)(\w+)',r'\1 \3',tweet)		# words(-|;)word separated
		tokens = tok.tokenize(tweet)
		
		pattern = re.compile(r"(.)\1{2,}", re.DOTALL) # hunggggryy -> hungryy

		newTokens = []
		flag = 0
		for word in tokens:
			word = pattern.sub(r"\1", word)
			# word = word.strip('\'"?,.!')
			word = word.strip('.,();-*~[]_=|+%')
			word = re.sub(r'(\w+)[..|.](\w+)',r'\1 \2',word)
			newWord = word.split()
			for word in newWord:
				word = wnl.lemmatize(word)
		# 		if(stopWords.has_key(word) or word == ''or word.isdigit())):
				if(stopWords.has_key(word) or word == '' or word.isdigit() or word=='\''):	
					continue
				else:
					if(flag == 1):
						word = "NOT_" + word
						flag = 0
					if(word == "n't" and flag == 0):
						flag = 1
						word = "not"
					newTokens.append(word)
		for word in newTokens:
			# str = str + word + ' '
			if(not wordListMap.has_key(word)):
				wordListMap[word] = 1
		if(len(newTokens)>0):
			totalTweets = totalTweets + 1
			tokenizedTweets.append(newTokens)

	classifier = pickle.load(maxEntObjectFile)
	for i in range(0,len(tokenizedTweets)):
		testTweet = tokenizedTweets[i]
		pred = classifier.classify(featureFunc(testTweet))
		outputFile.write(str(pred)+"\n")
	return

if __name__ == "__main__":
	main(sys.argv[1:])