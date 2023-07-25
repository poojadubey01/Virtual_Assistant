import speech_recognition as sr
import pyttsx3
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import datetime
import requests
import random
import wikipedia
import os
import webbrowser

class Chatbot:
    def __init__(self, training_data):
        self.training_data = training_data
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.classifier = MultinomialNB()
        self.labels = []
        self.responses = {}

    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return tokens

    def train(self):
        X_train = []
        y_train = []
        for data in self.training_data:
            X_train.append(data[0])
            y_train.append(data[1])
            if data[1] not in self.labels:
                self.labels.append(data[1])
        X_train = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train, y_train)

    def get_response(self, text):
        X_test = self.vectorizer.transform([text])
        y_pred = self.classifier.predict(X_test)[0]
        return self.responses[y_pred]

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except:
            self.speak("Sorry, I didn't catch that. Could you please repeat?")
            return self.listen()
            raise

    def get_time(self):
        now = datetime.datetime.now()
        time_string = now.strftime("%I:%M %p")
        return time_string

    def set_responses(self, responses):
        self.responses = responses

    def start(self):
        self.speak("Hi, I'm your personal assistant. How can I help you?")
        while True:
            text = self.listen()
            response = self.get_response(text)
            self.speak(response)

    # Define a function to fetch the summary from Wikipedia
    def get_wikipedia_summary(self, query):
        try:
            summary = wikipedia.summary(query)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options[:5]
            return f"Please be more specific. Did you mean {', '.join(options)}?"
        except wikipedia.exceptions.PageError:
            return "Sorry, I could not find any results for that search."

# Define a function to fetch the weather report from an API
def get_weather_report():
    api_key = "8ef32e48851da49dfea269407e7e5a67"
    city = "Mumbai"  # Replace with the desired city
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    temperature = data['main']['temp']
    description = data['weather'][0]['description']
    weather_report = f"The current temperature in {city} is {temperature} Kelvin with {description}."
    return weather_report

#function to fetch the quote from an API
def get_quote():
    url = "https://api.quotable.io/random"
    response = requests.get(url)
    data = response.json()
    quote = data['content']
    author = data['author']
    return f"{quote} by {author}"
    
# define the function to get a random joke
def get_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)
    data = response.json()
    joke = data['setup'] + " " + data['punchline']
    return joke

# define the function to get a random fact
def get_fact():
    facts = [
        "The world's largest pyramid is in Mexico and is over 2,300 years old.",
        "The shortest war in history was between Britain and Zanzibar in 1896. It lasted just 38 minutes.",
        "The world's oldest piece of chewing gum is over 9,000 years old.",
        "A small child could swim through the veins of a blue whale.",
        "The shortest commercial flight in the world is between two Scottish islands. It lasts just 1.7 miles and takes less than a minute.",
        "The only letter that doesn't appear in any US state name is 'Q'.",
        "The tallest mammal in the world is the giraffe, which can reach heights of up to 18 feet.",
        "A group of flamingos is called a flamboyance.",
        "The world's largest snowflake on record measured 15 inches wide and 8 inches thick.",
        "There are more possible iterations of a game of chess than there are atoms in the known universe.",
        "The world's oldest piece of cheese dates back to 1615 and is stored at the Museum of English Rural Life.",
        "Koalas have fingerprints that are almost indistinguishable from humans.",
        "The world's largest rubber duck is over 50 feet tall and weighs over 11 tons.",
        "The world's largest ball of twine weighs over 21,000 pounds and is over 12 feet wide.",
        "There are more lifeforms living on your skin than there are people on the planet.",
        "The world's largest jigsaw puzzle has over 551,232 pieces.",
        "A crocodile can't move its tongue and can't chew. Instead, it swallows stones to help grind up its food.",
        "The world's largest potato chip was over 23 inches wide and 14 inches long.",
        "The world's largest rubber band ball weighs over 9,000 pounds and is over 6 feet tall."
        "The world's largest ball of twine weighs over 21,000 pounds and is over 12 feet wide.",
    ]
    return random.choice(facts)

def main():
    training_data = [
        ('what is your name', 'name'),
        ('what is the time', 'time'),
        ('what is the weather like today', 'weather'),
        ('how old are you', 'age'),
        ('where are you from', 'location'),
        ('give me a quote', 'quote'),
        ('tell me a joke', 'joke'),
        ('tell me about', 'wikipedia'),
        ('tell me a fact', 'fact'),
        
    ]
    responses = {
        'greeting': 'Hello! How can I help you?',
        'name': "My name is Chatbot.",
        'time': f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.",
        'weather': "I'm fetching the weather information. Please wait for a moment.",
        'age': "I'm just a machine. I don't have an age.",
        'location': "I was created by Pooja DUbey.",
        'quote': get_quote(),
        'joke': get_joke(),
        'fact': get_fact(),
        'wikipedia': "I'm fetching the information from Wikipedia. Please wait for a moment.",
        'unknown': "Sorry, I didn't understand. Could you please repeat?"
    }
    # Update the weather response in the responses dictionary
    responses['weather'] = get_weather_report()
    
    chatbot = Chatbot(training_data)
    chatbot.set_responses(responses)
    chatbot.train()
    chatbot.start()

if __name__ == '__main__':
    main()

