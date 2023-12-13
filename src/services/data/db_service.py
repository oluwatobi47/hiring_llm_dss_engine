from sqlalchemy import create_engine, Engine


class SimpleSQLiteDbConnectionFactory:
    _connection_map = {}

    def create_connection(self, db_uri: str) -> Engine:
        if db_uri not in self._connection_map:
            self._connection_map[db_uri] = create_engine(db_uri)
        return self._connection_map[db_uri]


# Create shared instance of DB Connection factory to manage shared db engine instances across services
dbConnectionFactory = SimpleSQLiteDbConnectionFactory()


def load_db_client(db_uri: str) -> Engine:
    return dbConnectionFactory.create_connection(db_uri)
