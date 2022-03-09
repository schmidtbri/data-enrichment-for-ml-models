from typing import List
from pydantic import BaseModel, create_model
import psycopg2
from ml_base.decorator import MLModelDecorator


class PostgreSQLEnrichmentDecorator(MLModelDecorator):
    """Decorator to do data enrichment using a PostgreSQL database."""

    def __init__(self, host: str, port: str, username: str, password: str, database: str,
                 table: str, index_field_name: str, index_field_type: str,
                 enrichment_fields: List[str]) -> None:
        super().__init__(host=host, port=port, username=username, password=password,
                         database=database, table=table, index_field_name=index_field_name,
                         index_field_type=index_field_type, enrichment_fields=enrichment_fields)
        self.__dict__["_connection"] = None

    @property
    def input_schema(self) -> BaseModel:
        # converting the index field type from a string to a class
        index_field_type = getattr(__builtins__,
                                   self._configuration["index_field_type"])

        input_schema = self._model.input_schema

        # adding index field to schema because it is required in order to retrieve
        # the right record in the database
        fields = {
            self._configuration["index_field_name"]: (index_field_type, ...)
        }
        for field_name, schema in input_schema.__fields__.items():
            # remove enrichment_fields from schema because they'll be added from the
            # database and dont need to be provided by the client
            if field_name not in self._configuration["enrichment_fields"]:
                if schema.required:
                    fields[field_name] = (schema.type_, ...)
                else:
                    fields[field_name] = (schema.type_, schema.default)

        new_input_schema = create_model(
            input_schema.__name__,
            **fields
        )
        return new_input_schema

    def predict(self, data):
        # create a connection to the database, if it doesn't exist already
        if self.__dict__["_connection"] is None:
            self.__dict__["_connection"] = psycopg2.connect(
                host=self._configuration["host"],
                port=self._configuration["port"],
                database=self._configuration["database"],
                user=self._configuration["username"],
                password=self._configuration["password"])
        cursor = self.__dict__["_connection"].cursor()

        # build a SELECT statement using the index_field and the enrichment_fields
        enrichment_fields = ", ".join(self._configuration["enrichment_fields"])
        sql_statement = "SELECT {} FROM {} WHERE {} = %s;".format(
            enrichment_fields,
            self._configuration["table"],
            self._configuration["index_field_name"])

        # executing the SELECT statement
        cursor.execute(sql_statement,
                       (getattr(data, self._configuration["index_field_name"]), ))
        records = cursor.fetchall()
        cursor.close()

        if len(records) == 0:
            raise ValueError("Could not find a record for data enrichment.")
        elif len(records) == 1:
            record = records[0]
        else:
            raise ValueError("Query returned more than one record.")

        # creating an instance of the model's input schema using the fields that
        # came back from the database and fields that are provided by calling code
        input_schema = self.input_schema
        enriched_data = {}
        for field_name in self._model.input_schema.__fields__.keys():
            if field_name == self._configuration["index_field_name"]:
                pass
            elif field_name in self._configuration["enrichment_fields"]:
                field_index = self._configuration["enrichment_fields"].index(field_name)
                enriched_data[field_name] = record[field_index]
            elif field_name in data.dict().keys():
                enriched_data[field_name] = getattr(data, field_name)
            else:
                raise ValueError("Could not find value for field '{}'.".format(field_name))

        # making a prediction with the model, using the enriched fields
        enriched_data = self._model.input_schema(**enriched_data)
        prediction = self._model.predict(data=enriched_data)

        return prediction

    def __del__(self):
        if self._connection is not None:
            self._connection.close()
