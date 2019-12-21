import sqlite3
import pandas as pd

# db_path = 'db/test_records.db'
db_path = 'db/test.db'

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    sqlQuery = conn.cursor()
    # sqlQuery.execute("CREATE TABLE RECORDS \
    #           (ID INTEGER PRIMARY KEY AUTOINCREMENT,\
    #            num real NOT NULL,\
    #            x2 real NOT NULL,\
    #            x4 real NOT NULL,\
    #            temp real NOT NULL,\
    #             y1 real, y2 real,) ")

    sqlQuery.execute("CREATE TABLE RECORDS \
                  (\
                   num real NOT NULL,\
                   x2 real NOT NULL,\
                   x4 real NOT NULL,\
                   temp real NOT NULL,\
                    y1 real, y2 real) ")
    print("Create {0} db finish".format(db_path))

    conn.commit()
    conn.close()

def save_record(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    data = pd.read_csv('Ensemble/tiqu.csv')
    index = range(len(data))
    x_table = data.loc[:, ['Num', 'x2', 'x4', 'temp']]
    y_table = data.loc[:, ['y1', 'y2']]
    # X = x_table.iloc[index]
    #
    # Y = y_table.iloc[index]
    # print(x_table.iloc[0,0])
    for i in index:
        c.execute("INSERT INTO records (num,x2,x4,temp,y1,y2) VALUES "
                  "(?,?,?,?,?,?)",\
                  (x_table.iloc[i,0],x_table.iloc[i,1],\
                   x_table.iloc[i,2],x_table.iloc[i,3],\
                   y_table.iloc[i,0],y_table.iloc[i,1]))
    conn.commit()
    conn.close()


def View_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    cursor = c.execute("SELECT  num, x2, x4,temp,y1,y2  \
                                from records")
    for row in cursor:
        # print("ID = ", row[0])
        print("x = {},{},{},{}".format(row[0],row[1],row[2],row[3]))
        print("y1 ={0}, y2 = {1} ".format(row[4],row[5]), "\n")
    conn.close()



def delete_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    delete_len = 20
    # cursor.execute("DELETE FROM RECORDS WHERE id < "+ str(delete_len) )
    # 删除前 delete_len 记录
    tab_len = cursor.execute("SELECT count(*) from RECORDS")
    print(list(tab_len)[0], "  tab_len")
    # cursor.execute("DELETE FROM RECORDS WHERE id in (\
    #                 SELECT id from RECORDS order by id asc\
    #                  limit "+str(delete_len) +")")
    # cursor.execute("VACUUM  "+ db_path)# id 重新开始更新
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # create_db(db_path)
    # save_record(db_path)
    delete_db(db_path)
    View_db(db_path)