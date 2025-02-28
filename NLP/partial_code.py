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
