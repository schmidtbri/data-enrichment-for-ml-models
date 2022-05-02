import os
import unittest

from data_enrichment.postgresql import PostgreSQLEnrichmentDecorator


class DecoratorTests(unittest.TestCase):

    def test_init_password_value_replacement(self):
        # arrange
        os.environ["test"] = "asdf"

        # act
        decorator = PostgreSQLEnrichmentDecorator(host="str",
                                                  port="str",
                                                  username="str",
                                                  password="${test}",
                                                  database="str",
                                                  table="str",
                                                  index_field_name="str",
                                                  index_field_type="str",
                                                  enrichment_fields=[])

        # assert
        self.assertTrue(decorator._configuration["password"] == "asdf")


if __name__ == '__main__':
    unittest.main()
