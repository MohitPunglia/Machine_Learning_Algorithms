# %%
# import Spacy
#!pip install spacy

# %%
import spacy

# %%
nlp = spacy.load("en_core_web_sm")

# %%
!python3 -m spacy download en_core_web_sm


# %%
doc = nlp("Tesla is looking at buying U.S. startup for $6 million")

# %%
doc.text

# %%
doc[-1]

# %%
doc[0].pos_

# %%
doc[0].dep_

# %%
doc[0].tag_


# %%
spacy.explain("PROPN")

# %%
spacy.explain("NNP")

# %%
for word in doc:
    print(
        word.text,
        "---->",
        word.pos_,
        "---->",
        word.tag_,
        "---->",
        spacy.explain(word.tag_),
    )

# %%
doc2 = nlp("Tesla isn't looking into startups anymore.")
for word in doc2:
    print(
        word.text,
        "---->",
        word.pos_,
        "---->",
        word.tag_,
        "---->",
        spacy.explain(word.tag_),
    )

# %%
doc3 = nlp("A 5km NYC cab ride costs $10.30")
for word in doc3:
    print(
        word.text,
        "---->",
        word.pos_,
        "---->",
        word.tag_,
        "---->",
        spacy.explain(word.tag_),
    )
# %%
doc4 = nlp("Let's visit St. Louis in the U.S. next year.")

# %%
for word in doc4:
    print(
        word.text,
        "---->",
        word.pos_,
        "---->",
        word.tag_,
        "---->",
        spacy.explain(word.tag_),
    )

# %%
doc5 = nlp("India is a country in South Asia.")

# %%
for word in doc5:
    print(
        word.text,
        "---->",
        word.pos_,
        "---->",
        word.tag_,
        "---->",
        spacy.explain(word.tag_),
    )

# %%
doc6 = nlp("Tesla to build a U.K. factory for $6 million")
for word in doc6:
    print(
        word.text,
        "---->",
        word.pos_,
        "---->",
        word.tag_,
        "---->",
        spacy.explain(word.tag_),
    )
