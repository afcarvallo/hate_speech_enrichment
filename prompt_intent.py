# Dictionary for explainable descriptions of each category
Explainable_Abuse = {
    ### INTENT TAGS DESCRIPTIONS 
    }

def generate_ner_prompt(text, category):
    """
    Generates a prompt for an NER model to identify text spans showing a specified category of intent.

    Args:
    text (str): The text to be analyzed.
    category (str): The category to identify within the text.

    Returns:
    str: A formatted prompt instructing how to identify and label parts of the text.
    """
    if category not in Explainable_Abuse:
        raise ValueError(f"Category '{category}' is not supported. Choose from {list(Explainable_Abuse.keys())}.")

    prompt = (
        f"You are a Named Entity Recognition (NER) model. Your task is to identify spans of text that exhibit {category.lower()} based on the following definition:\n\n"
        f"{category}: {Explainable_Abuse[category]}\n\n"
        f"The provided text will probably contain comments that show {category.lower()}.\n\n"
        f"Text: \"{text}\"\n\nPlease identify the part of the text that show {category.lower()}. The span should contain the minimal amount of text in case of doubts whether to include a longer text fragment or not. Provide your response in the following format: '{category} span': 'text exhibiting {category.lower()}'"
    )
    return prompt