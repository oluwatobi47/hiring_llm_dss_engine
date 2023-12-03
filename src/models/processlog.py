
class ProcessLog:
    """
    A model class representing process logs with the execution time
    """
    def __init__(self, process_name: str, execution_time: int, comments: str = ""):
        """
        :param process_name: The process name
        :param execution_time: Time taken for process to complete
        :param comments: Comments on process (Optional)
        """
        self.__process_name = process_name
        self.__execution_time = execution_time
        self.__comments = comments

    def process_name(self):
        return self.__process_name

    def execution_time(self):
        return self.__execution_time

    def comments(self):
        return self.__comments

