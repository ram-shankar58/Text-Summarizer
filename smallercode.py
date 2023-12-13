import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample text to summarize
text = """This century is the age of Science. We cannot imagine our lives sans Science. Science has become a part and parcel of our lives. Science has become a symbol of progress. The progress is in the field of medicine, education, industry, etc., and we enjoy the comforts of science in all fields. Science has developed effective transport and communication system. Buses, cars, trains, planes have made transportation easy and comfortable, safe and fast. Man has even landed on moon with the help of technology.
In the field of medicine, science has worked wonders. Almost all kinds of diseases are entirely cured by modern drugs and medicines. Medicine has reduced pain and suffering. Electricity is another important scientific invention. the comforts of our life like electric lamps, refrigerators, fans, grinders, washing machines, etc. are all run by electricity.
Scientific method of cultivation has solved the flood problem. The pests destroying the crops are killed immediately by pesticides. Poultry and sericulture are also improved. Thus science is helpful in all walks of life and makes our life comfortable and happy"""
# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Tokenize the text into words and remove stopwords
stopWords = set(stopwords.words("english"))
words = word_tokenize(text)
freqTable = dict()
for word in words:
    word = word.lower()
    if word not in stopWords:
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

# Score sentences based on frequency of words
sentenceValue = dict()
for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

# Calculate the average score for sentences
sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]
average = int(sumValues / len(sentenceValue))

# Generate the summary
summary = ''
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
        summary += " " + sentence

print(summary)
