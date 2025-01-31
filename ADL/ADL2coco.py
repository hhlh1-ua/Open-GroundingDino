import os
import pandas as pd

HEIGTH = 960
WIDTH = 1280

categories = []  # Lista de categorías
images = []
category_names = set()  # Conjunto para verificar nombres únicos
images_names = set()
image_id_counter = 0
category_id_counter = 1  # Contador de ID autoincremental

def createCOCO():
    global category_id_counter  # Hacer el contador global para que persista
    global image_id_counter
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
                image_name= line_vals [5][2:]

                if image_name not in images_names:
                    image={
                        "id": image_id_counter,
                        "width": WIDTH,
                        "heigth": HEIGTH,
                        "file_name":image_name
                    }
                    images.append(image)
                    images_names.add(image_name)
                    image_id_counter+=1
                
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
    print(images)

def main(args=None):
    createCOCO()

if __name__ == '__main__':
    main()
