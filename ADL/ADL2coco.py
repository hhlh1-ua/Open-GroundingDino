import os

HEIGTH = 960
WIDTH = 1280

categories = []  # Lista de categorías con IDs
images = []  # Lista de imágenes con IDs
annotations = []  # Lista de anotaciones
category_names = {}  # Diccionario para asignar ID a categorías
images_dict = {}  # Diccionario para asignar ID a imágenes

image_id_counter = 0
category_id_counter = 1
annotation_id_counter = 1

def createCOCO():
    global category_id_counter, image_id_counter, annotation_id_counter
    
    for i in range(1, 7):
        filename = f'object_annot_P_0{i}.txt' if i < 10 else f'object_annot_P_{i}.txt'
        filedir = os.path.join('annotations', 'object_annotation', filename)
        
        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # Última palabra de la línea (categoría)
                image_name = line_vals[5][2:]  # Nombre de la imagen

                # Verificar si la imagen ya existe
                if image_name not in images_dict:
                    image_id_counter += 1
                    images_dict[image_name] = image_id_counter
                    images.append({
                        "id": image_id_counter,
                        "width": WIDTH,
                        "height": HEIGTH,
                        "file_name": image_name
                    })

                # Verificar si la categoría ya existe
                if category_name not in category_names:
                    category_names[category_name] = category_id_counter
                    categories.append({
                        "id": category_id_counter,
                        "name": category_name,
                        "supercategory": "none"
                    })
                    category_id_counter += 1

                # Coordenadas de la caja delimitadora (bbox)
                x1 = int(line_vals[1])
                y1 = int(line_vals[2])
                x2 = int(line_vals[3])
                y2 = int(line_vals[4])
                width = x2 - x1
                height = y2 - y1
                bbox = [x1, y1, width, height]

                # Crear la anotación
                annotation = {
                    "id": annotation_id_counter,
                    "image_id": images_dict[image_name],
                    "category_id": category_names[category_name],
                    "bbox": bbox,
                    "area": width * height,
                    "iscrowd": 0
                }
                annotations.append(annotation)
                annotation_id_counter += 1

    # print("Categorías:", categories)
    # print("Imágenes:", images)
    # print("Anotaciones:", annotations)

def main(args=None):
    createCOCO()

if __name__ == '__main__':
    main()
