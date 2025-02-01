import os
import json, glob
import argparse
ORIGINAL_HEIGTH = 960
ORIGINAL_WIDTH = 1280
### EN EL SERVIDOR TIENEN LAS IMÁGENES CON LA MITAD DE TAMAÑO QUE LO QUE PONE EN EL PAPER
HEIGTH = 480
WIDTH = 640



def create_label_map_extended(test, categories):

    if test:
        filename = "label_map_extended_test.json"
    else:
        filename = "label_map_extended_train.json"
    path = os.path.join('..','config', filename)

    inverted = {value: key for key, value in categories.items()}
    inverted_str = {str(key): inverted[key] for key in sorted(inverted.keys())}

    with open(path, 'w') as f:
        json.dump(inverted_str, f, indent=2)
    
 
    print(json.dumps(inverted_str, indent=2))


def createCOCO(outfile,start_range, fin_range):
    categories = []  # Lista de categorías con IDs
    images = []  # Lista de imágenes con IDs
    annotations = []  # Lista de anotaciones
    category_names = {}  # Diccionario para asignar ID a categorías
    images_dict = {}  # Diccionario para asignar ID a imágenes

    image_id_counter = 0
    category_id_counter = 0
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

                # Verificar si la imagen ya existe
                if image_name not in images_dict:
                    images_dict[image_name] = image_id_counter
                    images.append({
                        "id": image_id_counter,
                        "width": WIDTH,
                        "height": HEIGTH,
                        "file_name": os.path.join(prefix,image_name+'.jpg')
                    })
                    image_id_counter += 1

        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # Última palabra de la línea (categoría)
                image_name = line_vals[5][2:]  # Nombre de la imagen
                if image_name in images_dict:
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
                    x1 = float(line_vals[1]) / 2.0
                    y1 = float(line_vals[2]) / 2.0
                    x2 = float(line_vals[3]) / 2.0
                    y2 = float(line_vals[4]) / 2.0
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
    return category_names


def main(args):
    if(args.test):
        categories=createCOCO(outfile='annotations_test.json',start_range=11,fin_range=17)
        create_label_map_extended(True,categories)
    if(args.validation):
        createCOCO(outfile='annotations_val.json',start_range=17,fin_range=21)
    if(args.train):
       createCOCO(outfile='annotations_train.json',start_range=1,fin_range=11)
    if(args.all):
        createCOCO(outfile='annotations_train.json',start_range=1,fin_range=11)
        createCOCO(outfile='annotations_test.json',start_range=11,fin_range=17)
        createCOCO(outfile='annotations_val.json',start_range=17,fin_range=21)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decide what type of annotations create')
    parser.add_argument('-tst', '--test', help='Use to create test annotations', action='store_true')
    parser.add_argument('-trn', '--train', help='Use to create train annotations', action='store_true')
    parser.add_argument('-v', '--validation', help='Use to create validation annotations', action='store_true')
    parser.add_argument('-a', '--all', help='Use to create all annotations', action='store_true')
    main(parser.parse_args())
