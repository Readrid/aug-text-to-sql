import psycopg2
from psycopg2 import OperationalError

from database_connection.abstract_connector import AbstractDbConnector


class PostgresqlConnector(AbstractDbConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.connection = psycopg2.connect(**kwargs)
            print("Connection to PostgreSQL DB successful")
        except OperationalError as e:
            print(f"The error '{e}' occurred")

    def execute(self, query):
        self.connection.autocommit = True
        cursor = self.connection.cursor()

        query_result = None
        try:
            query_result = cursor.execute(query)
            print("Query executed successfully")
        except OperationalError as postgresql_error:
            print(f"The error '{postgresql_error}' occurred")

        return query_result
