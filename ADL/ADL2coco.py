import os
import pandas as pd

WIDTH = 960
HEIGTH = 1280

categories = []  # Lista de categorías
category_names = set()  # Conjunto para verificar nombres únicos
category_id_counter = 1  # Contador de ID autoincremental

def createCOCO():
    global category_id_counter  # Hacer el contador global para que persista
    for i in range(1,7):
        if(i<10):
            filename = f'object_annot_P_0{i}.txt'
        else:
            filename = f'object_annot_P_{i}.txt'
        filedir = os.path.join('annotations', 'object_annotation', filename)
        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # Última palabra de la línea
                
                if category_name not in category_names:
                    category = {
                        "id": category_id_counter,  # Se asigna el ID autoincremental
                        "name": category_name,
                        "supercategory": "none"
                    }
                    categories.append(category) 
                    category_names.add(category_name)  
                    
                    category_id_counter += 1  # Incrementar el contador para la siguiente categoría
    
    print(categories)  # Mostrar las categorías únicas con ID autoincremental

def main(args=None):
    createCOCO()

if __name__ == '__main__':
    main()
