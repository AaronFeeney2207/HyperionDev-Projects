import spacy 
nlp = spacy.load("en_core_web_sm")

# Storing all the sentences in a list
gardenpathSentences = ['I know the words to that song about the queen don’t rhyme.', 'The sour drink from the ocean', 'Mary gave the child a Band-Aid.', 
                        'That Jill is never here hurts.', 'The cotton clothing is made of grows in Mississippi.']

# Splitting each sentence intotheir own variables to be processed in order to tokenize the sentences
doc1 = nlp(gardenpathSentences[0])
doc2 = nlp(gardenpathSentences[1])
doc3 = nlp(gardenpathSentences[2])
doc4 = nlp(gardenpathSentences[3])
doc5 = nlp(gardenpathSentences[4])

# Tokenizing each sentence
doc1_tok = [token.orth_ for token in doc1]
print(doc1_tok)
doc2_tok = [token.orth_ for token in doc2]
print(doc2_tok)
doc3_tok = [token.orth_ for token in doc3]
print(doc3_tok)
doc4_tok = [token.orth_ for token in doc4]
print(doc4_tok)
doc5_tok = [token.orth_ for token in doc5]
print(doc5_tok)


# Named entity recognition on all sentences 
# 1st Senetence 
print([(y, y.label_, y.label) for y in doc1.ents])

# 2nd Sentence
print([(y, y.label_, y.label) for y in doc2.ents])

# 3rd Sentence
print([(y, y.label_, y.label) for y in doc3.ents])

# 4th Sentence 
print([(y, y.label_, y.label) for y in doc4.ents])

# 5th Sentence 
print([(y, y.label_, y.label) for y in doc5.ents])

# Explanations of entities

entity_gpe = spacy.explain("GPE")
print(f'GPE: {entity_gpe}')

entity_person = spacy.explain('PERSON')
print(f'PERSON: {entity_person}')

# Comments about entities:
# Did the entity make sense in terms of the word associated with it?
# For the entity 'PERSON' yes it made sense as the explanation was ' People, including fictional which makes sense
# 'GPE' meaning 'Countries, cities and states' was a bit more of a hidden one as GPE means geopolitical entity which links to a geographical area so then it makes more sense 
# but off first glance, makes less sense when in comparison to 'PERSON' and the explanation for that. 

