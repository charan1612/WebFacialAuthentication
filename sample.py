import sqlite3

def init_db():
    conn = sqlite3.connect('database/users.db')
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS users
              (id INTEGER PRIMARY KEY,
              username TEXT NOT NULL UNIQUE,
              face_data_path TEXT NOT NULL)
              ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
