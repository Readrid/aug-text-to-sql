import sqlite3
from sqlite3 import Error

from database_connection.abstract_connector import AbstractDbConnector


class SqliteConnector(AbstractDbConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.connection = sqlite3.connect(kwargs["path"])
            print("Connection to SQLite DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

    def execute(self, query):
        cursor = self.connection.cursor()
        query_result = None
        try:
            query_result = cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Error as sqlite_error:
            print(f"The error '{sqlite_error}' occurred")

        return query_result
