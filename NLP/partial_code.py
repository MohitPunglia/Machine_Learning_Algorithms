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