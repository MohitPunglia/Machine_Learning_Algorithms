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
