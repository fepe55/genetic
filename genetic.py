import numpy as np
import pygame


class Color:
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255, 45)

    BACKGROUND = pygame.Color(50, 50, 50)
    MALE = pygame.Color(149, 202, 255)
    FEMALE = pygame.Color(231, 84, 128)
    FOOD = pygame.Color(237, 41, 57)


class Cell:
    def __init__(self):
        self.fruits = []
        self.organisms = []

    def has_fruits(self):
        return not self.fruits == []

    def add_fruit(self, fruit):
        self.fruits.append(fruit)

    def remove_fruit(self, fruit):
        self.fruits.remove(fruit)

    def get_random_fruit(self):
        return np.random.choice(self.fruits)

    def has_organisms(self):
        return not self.organisms == []

    def add_organism(self, organism):
        self.organisms.append(organism)

    def remove_organism(self, organism):
        self.organisms.remove(organism)

    def get_random_organism(self):
        return np.random.choice(self.organisms)


class Environment:
    MIN_X = 0
    MAX_X = 100
    MIN_Y = 0
    MAX_Y = 50
    ORGANISM_SIZE = 10

    def __init__(self, organism_model, initial_population, initial_food):
        self.organism_model = organism_model
        self.population = []
        # self.space = np.empty([self.MAX_X, self.MAX_Y], dtype=list)
        self.space = []
        for x in range(self.MAX_X):
            row = []
            for y in range(self.MAX_Y):
                row.append(Cell())
            self.space.append(row)

        self.current_iteration = 0
        pygame.init()
        screen_x = self.MAX_X * self.ORGANISM_SIZE
        screen_y = self.MAX_Y * self.ORGANISM_SIZE
        self.screen = pygame.display.set_mode((screen_x, screen_y))

        for i in range(initial_food):
            self._add_food()

        for i in range(initial_population):
            organism = self.organism_model(environment=self)
            self._add_organism(organism)

    def _add_organism(self, organism):
        self.population.append(organism)
        self.space[organism.x][organism.y].add_organism(organism)

    def _remove_organism(self, organism):
        self.space[organism.x][organism.y].remove_organism(organism)
        self.population.remove(organism)

    def _move_organism(self, organism):
        self._remove_organism(organism)
        organism.move()
        self._add_organism(organism)

    def _battle(self, organisms):
        """
        A battle to the death. The loser gets removed from the population
        """
        strengths = [o.strength for o in organisms]
        position_of_the_strongest = strengths.index(max(strengths))

        winner = organisms[position_of_the_strongest]
        losers = list(filter(lambda x: x != winner, organisms))

        for organism in losers:
            self._remove_organism(organism)

    def _reproduce(self, organism1, organism2):
        if Organism.share_family(organism1, organism2):
            self._move_organism(organism1)
            self._move_organism(organism2)
        else:
            o = self.organism_model(
                environment=self, mother=organism1, father=organism2
            )
            self._add_organism(o)

    def _dispute(self, cell):
        """ A list of organisms, sharing the same space """
        organisms = cell.organisms
        if len(organisms) == 2:
            o1 = organisms[0]
            o2 = organisms[1]
            if o1.sex != o2.sex:
                self._reproduce(o1, o2)
            else:
                self._battle(organisms)
        else:
            self._battle(organisms)

    def _cell_color(self, cell):
        if all([organism.sex == 'M' for organism in cell.organisms]):
            return Color.MALE
        if all([organism.sex == 'F' for organism in cell.organisms]):
            return Color.FEMALE
        return Color.WHITE

    def draw_information(self):
        print('Iteration: {}. Population: {}'.format(
            self.current_iteration, len(self.population)
        ))

    def draw_space(self):
        self.screen.fill(Color.BACKGROUND)
        for x, row in enumerate(self.space):
            for y, cell in enumerate(row):
                if not cell.has_organisms():
                    continue
                color = Color.WHITE
                color = self._cell_color(cell)
                size = len(cell.organisms)
                center_point = (x * self.ORGANISM_SIZE, y * self.ORGANISM_SIZE)
                radius = self.ORGANISM_SIZE * size // 2
                pygame.draw.circle(
                    self.screen, color, center_point, radius, 0
                )
        pygame.display.flip()

    def draw(self):
        self.draw_information()
        self.draw_space()

    def iterate(self):
        self.current_iteration += 1
        """
        get two random organisms. With the possible outcomes
        0.- nothing happens
        1.- one kills the other
        2.- they mate and generate a new organism
        """

        for row in self.space:
            for cell in row:
                if not cell.has_organisms():
                    continue
                if len(cell.organisms) == 1:
                    self._move_organism(cell.get_random_organism())
                else:
                    self._dispute(cell)


class Organism:
    MUTATION_OPTIONS = [-1, 0, 1]
    MUTATION_PROBABILITIES = [0.4, 0.2, 0.4]
    FAMILY_GENERATIONS = 3
    MIN_SPEED = 1
    MAX_SPEED = 1
    MIN_STRENGTH = 0
    MAX_STRENGTH = 10

    def _random_attr(self):
        return np.random.choice(
            self.MUTATION_OPTIONS, p=self.MUTATION_PROBABILITIES
        )

    def _normalize(self, attr, min, max):
        return np.min([np.max([attr, min]), max])

    def __init__(self, environment, mother=None, father=None):
        if bool(mother) != bool(father):
            raise Exception

        random_speed = self._random_attr()
        random_strength = self._random_attr()
        random_x = self._random_attr()
        random_y = self._random_attr()

        self.mother = mother
        self.father = father
        self.environment = environment
        self.sex = np.random.choice(['M', 'F'])
        self.age = 0

        self.x = self.parents_x() + int(random_x)
        self.y = self.parents_y() + int(random_y)

        self.speed = self.parents_speed() + random_speed
        self.strength = self.parents_strength() + random_strength

        # Normalize everything between MIN and MAX
        self.speed = self._normalize(
            self.speed, self.MIN_SPEED, self.MAX_SPEED,
        )
        self.strength = self._normalize(
            self.strength, self.MIN_STRENGTH, self.MAX_STRENGTH,
        )
        self.x = self._normalize(
            self.x, self.environment.MIN_X, self.environment.MAX_X - 1,
        )
        self.y = self._normalize(
            self.y, self.environment.MIN_Y, self.environment.MAX_Y - 1,
        )

    @property
    def has_parents(self):
        return self.mother is not None and self.father is not None

    def parents_speed(self):
        """
        Returns parents attribute, or, if the organism doesn't have parents,
        return a random value
        """
        if self.has_parents:
            att = (self.mother.speed + self.father.speed) / 2
        else:
            att = np.random.rand() * self.MAX_SPEED
        return att

    def parents_strength(self):
        """
        Returns parents attribute, or, if the organism doesn't have parents,
        return a random value
        """
        if self.has_parents:
            att = (self.mother.strength + self.father.strength) / 2
        else:
            att = np.random.rand() * self.MAX_STRENGTH
        return att

    def parents_x(self):
        """
        Returns parents attribute, or, if the organism doesn't have parents,
        return a random value
        """
        if self.has_parents:
            att = self.mother.x
        else:
            att = int(np.random.rand() * self.environment.MAX_X)
        return att

    def parents_y(self):
        """
        Returns parents attribute, or, if the organism doesn't have parents,
        return a random value
        """
        if self.has_parents:
            att = self.mother.y
        else:
            att = int(np.random.rand() * self.environment.MAX_Y)
        return att

    @classmethod
    def share_family(cls, o1, o2):
        """True if they share family"""
        return bool(set(o1.family()) & set(o2.family()))

    @classmethod
    def _add_family_member(cls, organism, family_members=[], generation=0):
        """
        Helper method that recursively adds family members up until
        FAMILY_GENERATIONS generations
        """
        family_members.append(organism)
        if organism.has_parents and generation < cls.FAMILY_GENERATIONS:
            generation += 1
            family_members = cls._add_family_member(
                organism.mother, family_members, generation
            )
            family_members = cls._add_family_member(
                organism.father, family_members, generation
            )
        return family_members

    def family(self):
        """Returns a list of family members"""
        return self._add_family_member(self)

    def move(self):
        new_x = self.x + int(np.round(
            np.random.uniform(-self.speed, self.speed)
        ))
        new_y = self.y + int(np.round(
            np.random.uniform(-self.speed, self.speed)
        ))
        self.x = self._normalize(
            new_x, self.environment.MIN_X, self.environment.MAX_X - 1,
        )
        self.y = self._normalize(
            new_y, self.environment.MIN_Y, self.environment.MAX_Y - 1,
        )
        return self.x, self.y


if __name__ == '__main__':
    ITERATIONS = 100
    INITIAL_POPULATION = 30
    INITIAL_FOOD = 0
    FPS = 10

    environment = Environment(Organism, INITIAL_POPULATION, INITIAL_FOOD)
    clock = pygame.time.Clock()

    for i in range(ITERATIONS):
        environment.iterate()
        environment.draw_information()
        environment.draw_space()
        clock.tick(FPS)
