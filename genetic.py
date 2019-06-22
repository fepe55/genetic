import numpy as np


class Environment:
    INITIAL_POPULATION = 25

    def __init__(self, organism_model):
        self.organism_model = organism_model
        self.population = []
        self.current_iteration = 0

        for i in range(self.INITIAL_POPULATION):
            o = self.organism_model()
            self.population.append(o)

    def _battle(self, organism1, organism2):
        """
        A battle to the death. The loser gets removed from the population
        """
        strength1 = organism1.get_strength()
        strength2 = organism2.get_strength()

        if strength1 > strength2:
            self.population.remove(organism2)
        elif strength1 < strength2:
            self.population.remove(organism1)

    def _reproduce(self, organism1, organism2):
        o = self.organism_model(mother=organism1, father=organism2)
        self.population.append(o)

    def iterate(self):
        self.current_iteration += 1
        """
        get two random organisms. With the possible outcomes
        0.- nothing happens
        1.- one kills the other
        2.- they mate and generate a new organism
        """

        for i in range(20):
            o1, o2 = np.random.choice(self.population, 2, replace=False)

            random_number = np.random.rand()
            if random_number < 0.4:
                self._battle(o1, o2)
            elif random_number < 0.8:
                self._reproduce(o1, o2)


class Organism:
    MUTATION_OPTIONS = [-1, 0, 1]
    MUTATION_PROBABILITIES = [0.4, 0.2, 0.4]
    MIN_SPEED = 0
    MAX_SPEED = 10
    MIN_STRENGTH = 0
    MAX_STRENGTH = 10
    mother = None
    father = None

    def __init__(self, mother=None, father=None):
        random_speed = np.random.choice(
            self.MUTATION_OPTIONS, p=self.MUTATION_PROBABILITIES
        )
        random_strength = np.random.choice(
            self.MUTATION_OPTIONS, p=self.MUTATION_PROBABILITIES
        )

        if (mother and not father) or (father and not mother):
            raise Exception
        if mother:
            self.mother = mother
        if father:
            self.father = father

        self.speed = self.parents_speed() + random_speed
        self.strength = self.parents_strength() + random_strength

        # Normalize everything under MAX
        self.speed = np.min([
            np.max([self.speed, self.MIN_SPEED]),
            self.MAX_SPEED
        ])
        self.strength = np.min([
            np.max([self.strength, self.MIN_STRENGTH]),
            self.MAX_STRENGTH
        ])

    @property
    def has_parents(self):
        return self.mother is not None and self.father is not None

    def parents_speed(self):
        """
        Returns parents attribute, or, if the organism doesn't have parents,
        return a random value
        """
        if self.has_parents:
            att = (self.mother.get_speed() + self.father.get_speed()) / 2
        else:
            att = np.random.rand() * self.MAX_SPEED
        return att

    def parents_strength(self):
        """
        Returns parents attribute, or, if the organism doesn't have parents,
        return a random value
        """
        if self.has_parents:
            att = (self.mother.get_strength() + self.father.get_strength()) / 2
        else:
            att = np.random.rand() * self.MAX_STRENGTH
        return att

    def get_speed(self):
        return self.speed

    def get_strength(self):
        return self.strength


if __name__ == '__main__':
    ITERATIONS = 1000

    environment = Environment(organism_model=Organism)
    print(len(environment.population))
    avg_str = np.average([o.get_strength() for o in environment.population])
    print(avg_str)

    for i in range(ITERATIONS):
        environment.iterate()

    print(len(environment.population))
    avg_str = np.average([o.get_strength() for o in environment.population])
    print(avg_str)
