import os

import chromadb
from chromadb import Settings
from chromadb.api import ClientAPI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))


class SimpleVectorDBConnectionFactory:
    _connection_map = {}

    def create_connection(self, db_uri: str) -> ClientAPI:
        if db_uri not in self._connection_map:
            settings = Settings(
                persist_directory=db_uri,
                is_persistent=True,
            )
            settings.allow_reset = True
            self._connection_map[db_uri] = chromadb.PersistentClient(path=db_uri, settings=settings)
        return self._connection_map[db_uri]


# Create shared instance of Chroma Connection factory to manage shared db engine instances accross services
dbConnectionFactory = SimpleVectorDBConnectionFactory()


def load_chroma_client(db_uri: str = None) -> ClientAPI:
    """Loads instance of chroma db client in application, if db URI is not specified, application uses
     CHROMA_PATH value defined in the environment variables
     TODO: Check how single instance performs
    """
    db_uri = db_uri if db_uri is not None else os.getenv('CHROMA_PATH')
    return dbConnectionFactory.create_connection(db_uri)
