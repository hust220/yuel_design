import sqlite3
import duckdb
import os
import sys

def get_duckdb_type(sqlite_type):
    """将 SQLite 数据类型映射到 DuckDB 数据类型"""
    sqlite_type = sqlite_type.upper() if sqlite_type else ''
    if 'INT' in sqlite_type:
        return 'BIGINT'
    elif sqlite_type in ['TEXT', 'CHAR', 'CLOB']:
        return 'VARCHAR'
    elif sqlite_type in ['REAL', 'FLOAT', 'DOUBLE']:
        return 'DOUBLE'
    elif sqlite_type == 'BLOB':
        return 'BLOB'
    else:
        return 'VARCHAR'  # 默认类型

def convert_sqlite_to_duckdb(sqlite_path, duckdb_path):
    """执行数据库转换的主函数"""
    # 连接到 SQLite 数据库
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()
    
    # 连接到 DuckDB 数据库（自动创建新数据库）
    duckdb_conn = duckdb.connect(duckdb_path)
    
    # 获取所有用户表名（排除系统表）
    sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in sqlite_cur.fetchall()]
    
    for table in tables:
        print(f"正在转换表: {table}")
        
        # 获取列信息
        sqlite_cur.execute(f"PRAGMA table_info({table})")
        columns = sqlite_cur.fetchall()
        
        # 构建 CREATE TABLE 语句
        column_defs = []
        for col in columns:
            col_name = col[1]
            sqlite_type = col[2] or ''
            duckdb_type = get_duckdb_type(sqlite_type)
            column_defs.append(f'"{col_name}" {duckdb_type}')
        
        # 创建 DuckDB 表
        create_sql = f'CREATE TABLE "{table}" ({", ".join(column_defs)})'
        duckdb_conn.execute(create_sql)
        
        # 分批次插入数据
        sqlite_cur.execute(f'SELECT * FROM "{table}"')
        insert_sql = f'INSERT INTO "{table}" VALUES ({", ".join(["?"]*len(columns))})'
        
        batch_size = 1000  # 每批插入1000条记录
        while True:
            rows = sqlite_cur.fetchmany(batch_size)
            if not rows:
                break
            duckdb_conn.executemany(insert_sql, rows)
    
    # 关闭数据库连接
    sqlite_conn.close()
    duckdb_conn.close()

if __name__ == '__main__':
    # 输入输出配置
    input_db = sys.argv[1]
    output_db = sys.argv[2]
    
    # 如果目标文件已存在则删除
    if os.path.exists(output_db):
        os.remove(output_db)
    
    # 执行转换
    convert_sqlite_to_duckdb(input_db, output_db)
    print("数据库转换完成！")