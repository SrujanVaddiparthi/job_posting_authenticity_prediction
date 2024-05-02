import numpy as np
from geneticalgorithm import geneticalgorithm as ga

class GA:
    """
    Genetic Algorithm (GA) class for optimizing stuff. Uses the 'geneticalgorithm' library.
    """
    
    def __init__(self, function, bounds, dimension=2):
        """
        Set up the GA with a function you want to minimize, where it can play around,
        and how many variables it needs to think about.
        """
        self.function = function
        self.bounds = bounds
        self.dimension = dimension

    def run(self):
        """
        Let's run this genetic algorithm and get the best set of parameters for our function.
        """
        model = ga(
            function=self.function,
            dimension=self.dimension,
            variable_type='real',
            variable_boundaries=self.bounds
        )
        model.run()
        return model.output_dict['variable']

if __name__ == "__main__":
    # Testing the GA with a simple function: sum of squares.
    def test_function(X):
        """
        Just a simple function: it adds up the squares of whatever numbers you give it.
        """
        return np.sum(X**2)

    # Set up our GA with the test function and see what it finds.
    ga_instance = GA(function=test_function, bounds=np.array([[-10, 10], [-10, 10]]))
    best_params = ga_instance.run()
    print("Best parameters found:", best_params)
