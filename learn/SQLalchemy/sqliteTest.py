import pandas as pd
from sqlalchemy import create_engine

# Define engine and connect


def createsqlengine(_databasepath):
    # Create a SQL engine and return the object
    engine = create_engine('sqlite://' + _databasepath)
    print(engine.table_names())
    return engine


def createdbconnection(_engine):
    # Create a connection to use for interacting with the database
    con = _engine.connect()
    return con


def querydb(_querystring, _con):
    # send a query to an open connection
    rs = _con.execute(_querystring)
    return rs


def insertintotable(_con, _table, _values):
    # insert values into a specific table in a defined connection
    rows = len(_values)
    cols = len(_values[0])

    for row in range(rows):
        query = 'INSERT INTO ' + _table + ' VALUES ('
        rowdata = _values[row]
        for col in range(cols):
            query = query + rowdata[col]
        print(query)
        query = query + ')'
        print(query)
    _ = _con.execute(query)

    print('inserted ' + rows + ' rows of data')
    return True

