from PIL import Image
import os
import pandas as pd

def resize_images_and_csv(input_folder, output_folder, csv_path, output_csv_path, scale_factor=0.5):
    """
    Redimensionne les images et met à jour les coordonnées des bounding boxes dans le CSV.
    
    Args:
        input_folder (str): Dossier contenant les images d'origine.
        output_folder (str): Dossier où enregistrer les images redimensionnées.
        csv_path (str): Chemin du fichier CSV contenant les bounding boxes.
        output_csv_path (str): Chemin du CSV mis à jour.
        scale_factor (float): Facteur de réduction (ex: 0.5 pour diviser la taille par 2).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    
    # Mettre à jour les coordonnées et redimensionner les images
    for index, row in df.iterrows():
        filename = row['Image']
        img_path = os.path.join(input_folder, filename)
        output_img_path = os.path.join(output_folder, filename)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_resized.save(output_img_path)
            print(f"Image {filename} redimensionnée à {new_size}")
            
            # Mettre à jour les coordonnées
            df.at[index, 'xMin'] = int(row['xMin'] * scale_factor)
            df.at[index, 'yMin'] = int(row['yMin'] * scale_factor)
            df.at[index, 'width'] = int(row['width'] * scale_factor)
            df.at[index, 'height'] = int(row['height'] * scale_factor)
    
    # Sauvegarder le CSV mis à jour
    df.to_csv(output_csv_path, index=False)
    print(f"CSV mis à jour enregistré sous {output_csv_path}")

# Exemple d'utilisation
resize_images_and_csv("E:/Minecraft_AI/images", "E:/Minecraft_AI/images_reduc", "E:/Minecraft_AI/data.csv", "E:/Minecraft_AI/data_reduc.csv", 0.25)