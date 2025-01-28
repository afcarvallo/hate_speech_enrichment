# Dictionary for explainable descriptions of each intent category
Explainable_Abuse = {
    "Comparison": (
        "Comments that contain dehumanizing comparisons targeted to an individual or group based on their protected characteristics. "
        "These comments often include language that likens people to animals or objects, or suggests they are less than human. "
        "Examples include phrases like 'viruses,' 'disgusting animals,' or 'human monkeys.'"
    ),
    "Animosity": (
        "Comments that express implicit abusive language or subtle hostility towards individuals or groups based on their protected characteristics. "
        "These comments often include negative generalizations, stereotypes, or offensive remarks that are not overtly hateful but still convey disrespect or prejudice. "
        "Examples include mocking someone's culture or language, making derogatory insinuations about a group's capabilities or contributions, or expressing implicit bias in a condescending manner."
    ),
    "Threatening": (
        "Comments that contain explicit or implicit threats of violence or harm towards individuals or groups based on their protected characteristics. "
        "These comments often involve language that suggests physical violence, death, or severe harm, and can include direct threats, encouragement of violence, or expressing a desire to cause harm. "
        "Examples include statements like 'they should be shot,' 'burn all mosques,' or 'throw napalm at them.'"
    ),
    "Hatecrime": (
        "Comments that glorify, support, or deny hateful actions, events, organizations, and individuals. "
        "These comments often include expressions of admiration for, or agreement with, violent or hateful ideologies and actions, such as those of the Nazis or other extremist groups. "
        "Examples include statements like 'Nazis were actually very progressive,' 'Jews are responsible for every bad thing,' or 'more power to whites means more justice.'"
    ),
    "Derogation": (
        "Comments that contain derogatory terms or insults targeted to an individual or group based on their protected characteristics. "
        "These comments often include offensive language, name-calling, or negative stereotypes that demean or belittle the targeted individuals or groups. "
        "Examples include phrases like 'shitstains,' 'God's failure,' or 'tranny of the month,' as well as generalizations like 'Black folks are inherently inferior' or 'every single mussie is a resentful son of a bitch.'"
    )
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