import database

if __name__ == "__main__":
    database.db_init.db_init()
    tests.db_test.db_test()
    # TODO: Add data pipeline training function calls
    # TODO: Add model training function calls
    # TODO: Add model prediction function calls
    # TODO: Add reporting function calls
    # TODO: Add GUI training function calls
    database.db_close.db_close()
