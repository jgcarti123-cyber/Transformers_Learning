"""
A synthetic text corpus for training word2vec  (PROVIDED - read it, don't edit it).

Why synthetic? On real text you can't easily check whether the embeddings are "right".
Here we KNOW the hidden truth: every person word has three hidden attributes,

    gender: male / female        status: royal / common        age: adult / young

and each sentence reveals exactly ONE attribute through its context words, e.g.

    "the king sat on the throne"       -> reveals status = royal
    "the king said that he was tired"  -> reveals gender = male

The model is never told these attributes. If word2vec works, it should discover them
from co-occurrence alone, so that  king - man + woman  lands near  queen.
"""
import random

PEOPLE = {
    #            gender    status    age
    "king":     ("male",   "royal",  "adult"),
    "queen":    ("female", "royal",  "adult"),
    "prince":   ("male",   "royal",  "young"),
    "princess": ("female", "royal",  "young"),
    "man":      ("male",   "common", "adult"),
    "woman":    ("female", "common", "adult"),
    "boy":      ("male",   "common", "young"),
    "girl":     ("female", "common", "young"),
}

PERSON_TEMPLATES = {
    "male": ["the {w} said that he was tired", "the {w} lost his hat",
             "the {w} combed his hair", "he is the {w} we met"],
    "female": ["the {w} said that she was tired", "the {w} lost her hat",
               "the {w} combed her hair", "she is the {w} we met"],
    "royal": ["the {w} sat on the throne", "the {w} wore a golden crown",
              "the {w} lives in the palace", "the {w} rules the kingdom"],
    "common": ["the {w} works on the farm", "the {w} lives in the village",
               "the {w} paid the tax", "the {w} sells goods at the market"],
    "adult": ["the {w} drives to work", "the {w} reads the newspaper",
              "the {w} pays the rent", "the {w} votes in the election"],
    "young": ["the {w} plays with toys", "the {w} goes to school",
              "the {w} learns to read", "the {w} eats candy after class"],
}

ANIMALS = {"dog": "pet", "cat": "pet", "horse": "farm", "cow": "farm", "sheep": "farm"}
ANIMAL_TEMPLATES = {
    "any": ["the {w} has four legs", "the {w} drinks some water", "a {w} is an animal"],
    "pet": ["the {w} sleeps on the sofa", "the {w} chases the ball"],
    "farm": ["the {w} grazes in the meadow", "the {w} lives in the barn"],
}

FOODS = {"apple": "fruit", "banana": "fruit", "cherry": "fruit", "bread": "baked", "cake": "baked"}
FOOD_TEMPLATES = {
    "any": ["the {w} was delicious", "we ate the {w} for lunch", "the {w} costs one coin"],
    "fruit": ["the {w} grows on a tree", "the {w} is ripe and juicy"],
    "baked": ["the {w} came out of the oven", "the baker made the {w}"],
}

MIXED_TEMPLATES = ["the {p} fed the {a}", "the {p} ate the {f}", "the {p} walked with the {a}"]


def generate(n_sentences=30000, seed=0):
    """Returns a list of sentences, each a list of word strings."""
    rng = random.Random(seed)
    sentences = []
    for _ in range(n_sentences):
        r = rng.random()
        if r < 0.6:      # a sentence revealing one attribute of a person
            w = rng.choice(list(PEOPLE))
            attribute = rng.choice(PEOPLE[w])
            s = rng.choice(PERSON_TEMPLATES[attribute]).format(w=w)
        elif r < 0.75:
            w = rng.choice(list(ANIMALS))
            s = rng.choice(ANIMAL_TEMPLATES[rng.choice(["any", ANIMALS[w]])]).format(w=w)
        elif r < 0.9:
            w = rng.choice(list(FOODS))
            s = rng.choice(FOOD_TEMPLATES[rng.choice(["any", FOODS[w]])]).format(w=w)
        else:
            s = rng.choice(MIXED_TEMPLATES).format(
                p=rng.choice(list(PEOPLE)), a=rng.choice(list(ANIMALS)), f=rng.choice(list(FOODS)))
        sentences.append(s.split())
    return sentences


def build_vocab(sentences):
    """Returns (itos, stoi): a list of words and a dict word -> id, most frequent first."""
    counts = {}
    for s in sentences:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    itos = sorted(counts, key=lambda w: (-counts[w], w))
    return itos, {w: i for i, w in enumerate(itos)}


def category_of(word):
    if word in PEOPLE:
        return "person"
    if word in ANIMALS:
        return "animal"
    if word in FOODS:
        return "food"
    return "other"
