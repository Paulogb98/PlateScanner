import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.detector import YOLOv5Inference
from classes.recognizer import ONNXPlateRecognizer


def main():
    # Verificar se o modelo de detecção existe
    if not os.path.exists(model_path_detection):
        print(f"Erro: O arquivo {model_path_detection} não foi encontrado.")
        sys.exit(1)
    
    # Inicializar o modelo de detecção
    print("Inicializando o modelo de detecção...")
    yolo_detector = YOLOv5Inference(model_path_detection)
    
    # Processar as imagens de entrada (detecção de placas)
    print(f"Processando imagens do diretório: {input_images_dir}")
    yolo_detector.process_directory(input_images_dir, output_images_dir)
    
    # Verificar se o modelo de reconhecimento existe
    if not os.path.exists(model_path_recognition):
        print(f"Erro: O arquivo {model_path_recognition} não foi encontrado.")
        sys.exit(1)
    
    # Inicializar o modelo de reconhecimento
    print("Inicializando o modelo de reconhecimento...")
    recognizer = ONNXPlateRecognizer(model_path_recognition, config_path_recognition)
    
    # Processar as imagens cortadas (reconhecimento das placas)
    print(f"Processando imagens cortadas no diretório: {cropped_images_dir}")
    recognizer.process_cropped_images(cropped_images_dir, results_dir)
    
    print(f"Resultados salvos em: {results_dir}")


# Caminho dos modelos de detecção e reconhecimento
model_path_detection = '/app/models/detector/yolo_detector_model.pt'
config_path_recognition = "/app/models/recognizer/license_plates_ocr_config.yaml"
model_path_recognition = "/app/models/recognizer/license_plates_ocr_model.onnx"

# Caminho dos diretórios de entrada e saída
input_images_dir = '/app/data/raw'
output_images_dir = '/app/data/processed'
cropped_images_dir = '/app/data/processed/cropped'
results_dir = '/app/data/results'

if __name__ == "__main__":
    main()
