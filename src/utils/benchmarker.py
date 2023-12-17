import time


class Benchmarker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()

    def compute_execution_time(self):
        """
        Computes the execution time
        """
        self.execution_time = self.end_time - self.start_time

    def get_execution_time(self):
        """
        :return: The execution time taken for the process
        """
        return self.execution_time

    def benchmark_function(self, function, *args, **kwargs):
        """
        :param function: The function to be executed
        :param args: The function variable number of arguments
        :param kwargs: The function keyword arguments
        """
        self.start()
        output = function(*args, **kwargs)
        self.end()
        self.compute_execution_time()
        return output
