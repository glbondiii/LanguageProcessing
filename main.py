from secret import API_KEY
from data import obtain_data

import cohere
from cohere.responses.classify import Example
co = cohere.Client(API_KEY)

CLASSIFICATIONS = [
    "positive review",
    "neutral review",
    "negative review"
]

examples=[
    Example("The order came 5 days early", CLASSIFICATIONS[0]),
    Example("The item exceeded my expectations", CLASSIFICATIONS[0]),
    Example("I ordered more for my friends", CLASSIFICATIONS[0]),
    Example("I would buy this again", CLASSIFICATIONS[0]),
    Example("I would recommend this to others", CLASSIFICATIONS[0]),
    Example("This product is awesome", CLASSIFICATIONS[0]),
    Example("The package was damaged", CLASSIFICATIONS[2]),
    Example("The order is 5 days late", CLASSIFICATIONS[2]),
    Example("The order was incorrect", CLASSIFICATIONS[2]),
    Example("I want to return my item", CLASSIFICATIONS[2]),
    Example("The item's material feels low quality", CLASSIFICATIONS[2]),
    Example("The product was okay", CLASSIFICATIONS[1]),
    Example("I received five items in total", CLASSIFICATIONS[1]),
    Example("I bought it from the website", CLASSIFICATIONS[1]),
    Example("I used the product this morning", CLASSIFICATIONS[1]),
    Example("The product arrived yesterday", CLASSIFICATIONS[1])
]

def print_classifications(input):
    response = co.classify(
        inputs=input,
        examples=examples,
    )

    counts = [0] * len(CLASSIFICATIONS)
    totalCount = 0
    for classification in response:
        for i in range(len(CLASSIFICATIONS)):
            label = CLASSIFICATIONS[i]
            if classification.prediction == label:
                counts[i] += 1
                totalCount += 1
    
    for i in range(len(CLASSIFICATIONS)):
        print(f"{CLASSIFICATIONS[i]} {counts[i]/totalCount}")
    
inputs = obtain_data()
print_classifications(inputs)