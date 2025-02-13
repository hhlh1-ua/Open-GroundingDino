import os
import json, glob
import argparse
import re
ORIGINAL_HEIGTH = 960
ORIGINAL_WIDTH = 1280
### EN EL SERVIDOR TIENEN LAS IMÁGENES CON LA MITAD DE TAMAÑO QUE LO QUE PONE EN EL PAPER
HEIGTH = 480
WIDTH = 640
category_names = {}  # Diccionario para asignar ID a categorías
categories=[]


def create_label_map_extended(test, categories):

    filename = "label_map_ADL.json"

    path = os.path.join('..','config', filename)

    inverted = {value: key for key, value in categories.items()}
    inverted_str = {str(key): inverted[key] for key in sorted(inverted.keys())}

    with open(path, 'w') as f:
        json.dump(inverted_str, f, indent=2)
    


def update_coco2ovdgADL(categories):
    coco2ovdg_ADL_file = os.path.join('..','tools','coco2odvg_ADL.py')
    new_id_map = {value: value for value in categories.values()}  # Cada ID mapea a sí mismo
    new_id_map = str(new_id_map)  # Convertir a cadena, si es necesario


    inverted = {value: key for key, value in categories.items()}
    inverted_str = {str(key): inverted[key] for key in sorted(inverted.keys())}
    inverted_str = json.dumps(inverted_str, indent=4)  # Formatear como JSON con comillas dobles
    # Leer el contenido del archivo
    with open(coco2ovdg_ADL_file, 'r') as file:
        content = file.read()

    # Reemplazar id_map en el archivo
    content = re.sub(r'id_map\s*=\s*\{[^\}]*\}', f'id_map = {new_id_map}', content)

    # Reemplazar ori_map en el archivo
    content = re.sub(r'ori_map\s*=\s*\{[^\}]*\}', f'ori_map = {inverted_str}', content)

    # Escribir el contenido actualizado de vuelta al archivo
    with open(coco2ovdg_ADL_file, 'w') as file:
        file.write(content)


def updat_cfg(file,categories):
    sorted_categories = sorted(categories.items(), key=lambda x: x[1])
    label_list = [cat_name for cat_name, cat_id in sorted_categories]
    
    new_label_list_str = json.dumps(label_list)
    

    with open(file, 'r') as f:
        content = f.read()

    pattern = r'(label_list\s*=\s*)\[[^\]]*\]'
    

    new_content = re.sub(pattern, r'\1' + new_label_list_str, content, flags=re.DOTALL)
    
    with open(file, 'w') as f:
        f.write(new_content)


def getCategories():
    global category_names
    global categories
    category_id_counter = 0
    for i in range(1, 21):
        filename = f'object_annot_P_0{i}.txt' if i < 10 else f'object_annot_P_{i}.txt'
        filedir = os.path.join('annotations', 'object_annotation', filename)
        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # Última palabra de la línea (categoría)
                                # Verificar si la categoría ya existe
                if category_name not in category_names:
                    category_names[category_name] = category_id_counter
                    categories.append({
                        "id": category_id_counter,
                        "name": category_name,
                        "supercategory": "none"
                    })
                    category_id_counter += 1

def createCOCO(outfile,start_range, fin_range):
    images = []  # Lista de imágenes con IDs
    annotations = []  # Lista de anotaciones
    
    images_dict = {}  # Diccionario para asignar ID a imágenes

    image_id_counter = 0
    
    annotation_id_counter = 1
    errores=[]
    
    for i in range(start_range, fin_range):
        filename = f'object_annot_P_0{i}.txt' if i < 10 else f'object_annot_P_{i}.txt'
        filedir = os.path.join('annotations', 'object_annotation', filename)
        annotated_frames=f'object_annot_P_0{i}_annotated_frames.txt' if i < 10 else f'object_annot_P_{i}_annotated_frames.txt'
        filedir_annotated_frames = os.path.join('annotations', 'object_annotation', annotated_frames)
        prefix=f'P_0{i}'if i < 10 else f'P_{i}'
        print(f'getting annotations for frames in video {prefix}')
        with open(filedir_annotated_frames, "r") as file:
            for line in file:
                line_vals = line.split()
                image_name = line_vals[0][2:]  # Nombre de la imagen
                image_name=os.path.join(prefix,image_name+'.jpg')

                # Verificar si la imagen ya existe
                if image_name not in images_dict:
                    images_dict[image_name] = image_id_counter
                    images.append({
                        "id": image_id_counter,
                        "width": WIDTH,
                        "height": HEIGTH,
                        "file_name": image_name
                    })
                    image_id_counter += 1

        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # Última palabra de la línea (categoría)
                image_name = line_vals[5][2:]  # Nombre de la imagen
                image_name=os.path.join(prefix,image_name+'.jpg')
                if image_name in images_dict:


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
                        "segmentation": [],
                        "area": width * height,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "ignore": 0
                    }
                    annotations.append(annotation)
                    annotation_id_counter += 1
                else:
                    error={
                        "video":f'P{i}',
                        "frame": image_name
                    }
                    errores.append(error)
    coco={
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    outdir= os.path.join('annotations','coco_format',outfile)
    with open(outdir, 'w') as out_file:
        json.dump(coco, out_file, indent=4)


def main():
    getCategories()
    createCOCO(outfile='annotations_train.json',start_range=1,fin_range=5)
    update_coco2ovdgADL(category_names)
    create_label_map_extended(test=False,categories=category_names)
    createCOCO(outfile='annotations_test.json',start_range=7,fin_range=21)
    create_label_map_extended(test=True,categories=category_names)
    createCOCO(outfile='annotations_val.json',start_range=5,fin_range=7)
    file=os.path.join('..','config','cfg_ADL.py')
    updat_cfg(file,category_names)




if __name__ == '__main__':
    main()
