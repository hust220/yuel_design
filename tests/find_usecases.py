import psycopg2
from rdkit import Chem
import os
from io import StringIO

# 数据库连接配置
DB_CONFIG = {
    'dbname': 'moad',
    'user': 'juw1179',  # 替换为你的数据库用户名
    'host': 'submit03',  # 替换为你的数据库主机
    'port': '5433'  # 替换为你的数据库端口
}

# 输出目录配置
OUTPUT_DIR = '5izj_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_and_save_ligands():
    """获取并保存所有protein_name为5izj的配体"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 查询所有protein_name为5izj的配体
        cursor.execute(
            "SELECT name, mol FROM ligands WHERE protein_name = '5izj'"
        )
        
        ligands = cursor.fetchall()
        print(f"找到 {len(ligands)} 个配体")
        
        for name, mol_bytes in ligands:
            try:
                # 将二进制数据转换为RDKit分子对象
                mol = Chem.MolFromMolBlock(mol_bytes.tobytes().decode('utf-8'))
                if mol:
                    # 保存为.mol文件
                    output_path = os.path.join(OUTPUT_DIR, f"{name}.mol")
                    Chem.MolToMolFile(mol, output_path)
                    print(f"成功保存配体: {output_path}")
                else:
                    print(f"无法解析配体: {name}")
            except Exception as e:
                print(f"处理配体 {name} 时出错: {str(e)}")
        
    except Exception as e:
        print(f"数据库查询出错: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def fetch_and_save_protein():
    """获取并保存5izj蛋白结构"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 查询protein_name为5izj的蛋白结构
        cursor.execute(
            "SELECT pdb FROM proteins WHERE name = '5izj' LIMIT 1"
        )
        
        result = cursor.fetchone()
        if result:
            pdb_data = result[0].tobytes().decode('utf-8')
            
            # 保存为.pdb文件
            output_path = os.path.join(OUTPUT_DIR, "5izj.pdb")
            with open(output_path, 'w') as f:
                f.write(pdb_data)
            print(f"成功保存蛋白结构: {output_path}")
        else:
            print("未找到protein_name为5izj的蛋白记录")
        
    except Exception as e:
        print(f"数据库查询出错: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("开始处理5izj数据...")
    fetch_and_save_ligands()
    fetch_and_save_protein()
    print("处理完成！结果保存在目录:", os.path.abspath(OUTPUT_DIR))