import mysql.connector
from mysql.connector import Error

from database_connection.abstract_connector import AbstractDbConnector


class MysqlConnector(AbstractDbConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.connection = mysql.connector.connect(**kwargs)
            print("Connection to MySQL DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

    def execute(self, query):
        cursor = self.connection.cursor()
        query_result = None
        try:
            query_result = cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Error as mysql_error:
            print(f"The error '{mysql_error}' occurred")

        return query_result
