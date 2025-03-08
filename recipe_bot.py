import rasa
import spacy
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import json


nlp = spacy.load("en_core_web_sm")


recipes = {
    "chicken garlic": "Garlic Butter Chicken: Start by seasoning chicken breasts with salt and pepper. In a pan, melt butter and sauté minced garlic until fragrant. Add the chicken and cook until golden brown on both sides. Garnish with freshly chopped parsley and serve with a side of steamed vegetables or mashed potatoes."
}

class ActionProvideRecipe(Action):
    def name(self):
        return "action_provide_recipe"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")
        doc = nlp(user_input)
        ingredients = " ".join([token.text.lower() for token in doc if token.pos_ == "NOUN"])
        
        recipe = recipes.get(ingredients, "Sorry, I couldn't find a recipe with those ingredients.")
        
        dispatcher.utter_message(text=recipe)
        return []


def train_model():
    rasa.train(domain="domain.yml", config="config.yml", training_files="data")

if __name__ == "__main__":
    train_model()
