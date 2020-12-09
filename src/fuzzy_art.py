from dataclasses import dataclass
import logging as log
import numpy as np

class FuzzyArt:
    def __init__(self, pattern_size: int, vigilance: float, choice: float, learn_rate: float):
        self.pattern_size = pattern_size
        self.vigilance = vigilance
        self.choice = choice
        self.learn_rate = learn_rate
        self.categories = []
        self.category_counts = []

    def train(self, pattern: np.array):
        # check that pattern is the correct length
        if pattern.size != self.pattern_size:
            log.warn("input was the wrong size")
            return None

        # select the winner category and learn the pattern with it
        J = self.choose_category(pattern)
        self.learn_pattern(J, pattern)

        # return the category as the label
        return J

    def choose_category(self, pattern: np.array):
        N = len(self.categories)
        memberships = np.zeros(N)
        choices = np.zeros(N)

        # find the choice for each category
        for j in range(0, N):
            category = self.categories[j]
            fuzzy_and = np.minimum(pattern, category)
            numer = np.linalg.norm(fuzzy_and, 1)
            denom = self.choice + np.linalg.norm(category, 1)

            memberships[j] = numer
            choices[j] = numer / denom

        # iterate through categories by descending choice
        # until we find one that meets vigilance criteria
        pattern_norm = np.linalg.norm(pattern, 1)
        order = np.argsort(choices)
        for i in range(0, len(order)):
            j = order[i]
            match = memberships[j] / pattern_norm
            if match >= self.vigilance: return j

        # none of the categories matched
        # add a new one
        self.categories.append(np.ones(self.pattern_size))
        self.category_counts.append(0)
        return N

    def learn_pattern(self, J: int, pattern: np.array):
        category = self.categories[J]
        category = self.learn_rate * np.minimum(pattern, category) + (1 - self.learn_rate) * category
        self.categories[J] = category
        self.category_counts[J] += 1

def complement_code(x: np.array):
    return np.concatenate((x, 1 - x))

if __name__ == "__main__":
    # instantiate the ART module
    art = FuzzyArt(
        pattern_size=4,
        vigilance=0.9,
        choice=0.9,
        learn_rate=0.5
    )

    # present some random patterns
    for i in range(0, 10):
        pattern = np.array(np.random.rand(art.pattern_size // 2, 1))
        pattern_coded = complement_code(pattern)
        label = art.train(pattern_coded)
        print(f"{label:2}: {pattern.T}")
