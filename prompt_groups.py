def generate_ner_prompt(text, entities) -> str:
    """
    Generates a prompt for tagging named entities in a text that contains hate speech,
    focusing on two specific categories of entities.

    Parameters:
    ----------
    text : str
        The input text which needs to be processed for entity recognition.
    entities : str
        A formatted string of entities to be identified in the text.

    Returns:
    -------
    str
        A formatted string prompt instructing how to tag the entities in the provided text.
    """
    prompt = (
        f"""You are provided with a text that contains hate speech. Your task is to identify and tag all named entities within the text using the following entity types only, which fall into two categories:

1. Group Entities (prefix "G-"): These entities refer to specific groups targeted by hate speech.
2. Offensive Denominations (prefix "OD-"): These are derogatory terms or phrases used to insult or demean specific groups of people. These terms often carry negative connotations and are intended to belittle or dehumanize the group they target.

{entities}

You must provide the following information for each entity:
- The tag of the entity (uppercase);
- The value of the entity (as it appears in the text).

Your response should be JSON formatted using the following schema:

{{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "array",
  "items": [
    {{
      "type": "object",
      "properties": {{
        "tag": {{
          "type": "string"
        }},
        "value": {{
          "type": "string"
        }}
      }},
      "required": ["tag", "value"]
    }}
  ]
}}


Input:
{text}

Output JSON:
""")
    return prompt

def re_naive_json_extractor(json_string: str, expected_output: str = "object") -> str:
    """
    Extracts the first JSON-like object or array found in a string using regular expressions.

    Parameters:
    ----------
    json_string : str
        The string to search within for a JSON object or array.

    expected_output : str, optional
        Specifies whether to return an empty JSON object or array if no JSON is found. 
        Defaults to "object".

    Returns:
    -------
    str
        A JSON string if found, otherwise an empty JSON object or array based on expected_output.
    """
    json_pattern = r'(\{.*\}|\[.*\])'
    match = re.search(json_pattern, json_string, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return "{}" if expected_output == "object" else "[]"

# Function to detect hate speech entities using the OpenAI API
def detect_hate(text, category):
    """
    Detects hate speech entities in a given text using a model from OpenAI.

    Parameters:
    ----------
    text : str
        The text to analyze for hate speech entities.
    category : str
        The specific categories of hate speech to detect.

    Returns:
    -------
    str
        The response from the model containing detected entities.
    """
    prompt = generate_ner_prompt(text, category)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5,
        top_p=1,
    )
    return response.choices[0].message.content

GROUP_MENTIONS = {
    "G-INTERSECTION": "Entities representing the intersection of two or more identity groups (gender, sexual orientation, religion, ethnicity, disability, working class, ideological group). This category recognizes individuals belonging to multiple marginalized groups simultaneously, facing unique forms of discrimination. Examples include Black women, Muslim women, Asian women, gay people of color, or immigrants with disabilities. Insults targeting these intersectional identities often combine stereotypes from multiple aspects of a person's identity.",

    "G-GENDER": "Entities directly attacked or targeted with hate speech based on gender: women, men, transgender individuals, non-binary people, gender-fluid individuals, intersex people, cisgender people when explicitly targeted, gender non-conforming individuals when attacked, agender persons facing discrimination, bigender persons experiencing hate, two-spirit individuals subjected to prejudice, gender minorities when targeted, feminists or masculinists when attacked for their views",

    "G-SEXUAL_ORIENTATION":"Entities directly or indirectly attacked or targeted with hate speech based on sexual orientation: gay individuals facing homophobia, lesbian individuals subjected to discrimination, bisexual people experiencing biphobia, queer individuals targeted for their identity, pansexual people facing prejudice, asexual individuals subjected to erasure or attack, heterosexual people when explicitly targeted (rare contexts), aromantic persons experiencing discrimination, demisexual individuals facing prejudice, sexual minorities when attacked",

    "G-RELIGION": "Entities directly or indirectly attacked with hate speech based on religion: Muslims facing Islamophobia, Jews subjected to antisemitism, Christians when targeted for their faith, Hindus experiencing religious discrimination, Sikhs subjected to prejudice, Buddhists facing religious intolerance, atheists targeted for their lack of belief, agnostics when attacked for their views, Jains, Taoists, Wiccans, or Pagans subjected to religious discrimination, Scientologists when targeted, Mormons facing prejudice, Jehovah's Witnesses experiencing discrimination, religious converts subjected to attack, apostates facing threats or hate, spiritual but not religious individuals when targeted, followers of indigenous religions experiencing prejudice, religious minorities subjected to intolerance",

    "G-ETHNICITY": "Entities directly or indirectly attacked or targeted with hate speech based on ethnicity or race: Black people subjected to racism, Asian people facing discrimination, Latino people experiencing prejudice, indigenous people subjected to racism, racial/ethnic minorities when targeted, White people in contexts of explicit reverse discrimination, Arab people facing prejudice, Pacific Islanders subjected to racism, mixed-race individuals experiencing discrimination, specific ethnic groups (e.g., Uyghurs, Rohingya, Roma) when targeted, immigrants or refugees subjected to xenophobia or used as proxies for ethnic attacks",

    "G-DISABILITY": "Entities directly or indirectly attacked or targeted with hate speech based on disabilities: people with physical disabilities facing discrimination, individuals with intellectual disabilities subjected to prejudice, persons with learning disabilities when targeted, people with mental health conditions experiencing stigma or attack, neurodivergent individuals (e.g., autistic people, individuals with ADHD) facing prejudice, people with chronic illnesses subjected to discrimination, individuals with sensory disabilities (e.g., blind or deaf people) when targeted, people with invisible disabilities facing prejudice or erasure, disability rights activists subjected to attack for their advocacy",

    "G-WORKING_CLASS": "Entities directly or indirectly referenced or targeted based on socio-economic status, particularly those identified as part of the working class. This category includes individuals from lower-income groups who may be mentioned in the context of labor rights, economic struggles, or class-based discrimination. Examples include references to union members, blue-collar workers, and individuals from economically disadvantaged backgrounds, especially when their socio-economic status is highlighted in discussions or criticisms.",

    "G-IDEOLOGICAL_GROUP": "Entities directly or indirectly referenced or targeted based on political or ideological beliefs. This category encompasses political parties, movements, or belief systems that are mentioned in the context of ideological discussions or conflicts. Examples include references to communists, fascists, socialists, liberals, conservatives, anarchists, or members of any other political ideology or movement, particularly when these beliefs are central to the context of the mention or criticism."
}

OFFENSIVE_DENOMINATION = {
    "OD-INTERSECTION": "An insult that targets multiple identity factors simultaneously: slurs or derogatory terms that combine stereotypes related to race, gender, and socioeconomic status, such as 'welfare queen' or 'illegal alien', crude insults that intersect various marginalized identities, often amplifying the harm by drawing on multiple prejudices at once.",

    "OD-GENDER": "An insult based on gender: misogynistic slurs for women, derogatory terms for men, offensive labels for transgender individuals, insulting words for non-binary people, crude terms related to gender expression",

    "OD-SEXUAL_ORIENTATION": "An insult based on sexual orientation: homophobic slurs, offensive words for gay or lesbian individuals, derogatory terms for bisexual people, insulting labels for queer individuals, crude words targeting asexual or pansexual people",

    "OD-RELIGION": "An insult based on religion: anti-Muslim slurs, antisemitic insults, offensive terms for Christians, derogatory labels for Hindus or Sikhs, insulting words for Buddhists or atheists, crude terms for religious practices or beliefs",

    "OD-ETHNICITY": "An insult based on ethnicity or race: racial slurs for various ethnic groups, offensive terms based on skin color, insulting words for immigrants or refugees, derogatory labels for specific nationalities, crude ethnic stereotypes expressed as insults",

    "OD-DISABILITY":"An insult targeting perceived mental or physical capabilities: ableist slurs, offensive words for physical disabilities, derogatory terms for mental health conditions, insulting labels for neurodivergences, crude words implying lack of intelligence or mental capacity (e.g., 'idiot', 'stupid', 'dumb', 'retarded'), animal-based insults implying low intelligence (e.g., 'donkey')",

    "OD-WORKING_CLASS": "An insult based on socioeconomic status or class background: derogatory terms that demean individuals from lower economic classes, such as 'trailer trash' or 'redneck', crude labels implying poverty or lack of sophistication, offensive words targeting working-class individuals or communities.",

    "OD-IDEOLOGICAL_GROUP": "An insult based on political or ideological beliefs: pejorative terms used to describe political or ideological groups, such as 'commie' for communists, 'fascist' used inappropriately, or other derogatory labels for political ideologies, movements, or parties, especially when used to dismiss or belittle someone’s beliefs.",

}