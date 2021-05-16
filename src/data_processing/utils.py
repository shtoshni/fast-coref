def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def flatten(l):
  return [item for sublist in l for item in sublist]

