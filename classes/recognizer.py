import onnxruntime as ort
import numpy as np
from typing import List, Union, Tuple
import numpy.typing as npt
import os
import cv2
import yaml
import csv
import traceback


class ONNXPlateRecognizer:
    def __init__(self, model_path: str, config_path: str):
        """
        Inicializa o reconhecedor de placas com o modelo ONNX e configurações definidas em YAML.
        """
        self.model_path = model_path
        self.config = self.load_config(config_path)
        self.model = self.load_model(model_path)

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Carrega configurações do arquivo YAML.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração '{config_path}' não encontrado!")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_model(model_path: str) -> ort.InferenceSession:
        """
        Carrega o modelo ONNX, utilizando GPU se disponível.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo '{model_path}' não encontrado!")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=providers)

    @staticmethod
    def read_plate_image(image_path: str) -> npt.NDArray:
        """
        Lê uma imagem em escala de cinza a partir do caminho fornecido.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo '{image_path}' não encontrado!")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Falha ao carregar a imagem '{image_path}'.")
        return img

    def _load_image_from_source(
        self, source: Union[str, List[str], npt.NDArray, List[npt.NDArray]]
    ) -> Union[npt.NDArray, List[npt.NDArray]]:
        """
        Carrega imagem(ns) de uma fonte que pode ser caminho(s) ou array numpy.
        """
        if isinstance(source, str):
            return self.read_plate_image(source)

        if isinstance(source, list):
            if all(isinstance(s, str) for s in source):
                return [self.read_plate_image(i) for i in source]
            if all(isinstance(a, np.ndarray) for a in source):
                return source
            raise ValueError("Lista deve conter apenas `str` ou `np.ndarray`.")

        if isinstance(source, np.ndarray):
            source = source.squeeze()
            if source.ndim != 2:
                raise ValueError("Array deve ter forma (H, W) ou (H, W, 1).")
            return source

        raise TypeError("Tipo de entrada não suportado. Forneça caminho ou numpy array.")

    def preprocess_image(self, image: npt.NDArray, img_height: int, img_width: int) -> npt.NDArray:
        """
        Pré-processa a(s) imagem(ns) para o modelo.
        """
        if isinstance(image, np.ndarray):
            image = [image]

        imgs = np.array([
            cv2.resize(im.squeeze(), (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            for im in image
        ])
        imgs = np.expand_dims(imgs, axis=-1)  # Adiciona canal único (grayscale)
        imgs = imgs.astype(np.uint8)  # Normaliza a imagem
        return imgs

    def postprocess_output(
        self, model_output: npt.NDArray, max_plate_slots: int, model_alphabet: str,
        pad_char: str, return_confidence: bool = False
    ) -> Union[List[str], Tuple[List[str], npt.NDArray]]:
        """
        Converte a saída do modelo em texto (placas) e retorna as probabilidades se solicitado.
        """
        predictions = model_output.reshape((-1, max_plate_slots, len(model_alphabet)))
        prediction_indices = np.argmax(predictions, axis=-1)
        alphabet_array = np.array(list(model_alphabet))
        plate_chars = alphabet_array[prediction_indices]

        # Remove caracteres de padding
        plates = [''.join(plate).replace(pad_char, '') for plate in plate_chars]

        if return_confidence:
            probs = np.max(predictions, axis=-1)
            return plates, probs
        return plates

    def run(
        self,
        source: Union[str, List[str], npt.NDArray, List[npt.NDArray]],
        return_confidence: bool = False,
    ) -> Union[List[str], Tuple[List[str], npt.NDArray]]:
        """
        Executa o OCR para reconhecer caracteres da placa.
        """
        try:
            # Carregar e pré-processar
            x = self._load_image_from_source(source)
            x = self.preprocess_image(x, self.config["img_height"], self.config["img_width"])

            # Inferência
            y: List[npt.NDArray] = self.model.run(None, {"input": x})

            # Pós-processamento
            return self.postprocess_output(
                y[0],
                self.config["max_plate_slots"],
                self.config["alphabet"],
                self.config["pad_char"],
                return_confidence=return_confidence,
            )
        except Exception as e:
            print(f"Erro durante o processamento: {e}")
            traceback.print_exc()
            return [], None if return_confidence else []

    def process_cropped_images(self, cropped_images_dir: str, results_dir: str):
        """
        Processa todas as imagens cortadas e salva os resultados em CSV.
        """
        if not os.path.exists(cropped_images_dir):
            raise FileNotFoundError(f"Diretório '{cropped_images_dir}' não existe.")

        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'ocr_results.csv')

        with open(results_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Image Name', 'Extracted Value'])  # Cabeçalho

            for img_name in os.listdir(cropped_images_dir):
                img_path = os.path.join(cropped_images_dir, img_name)
                if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ocr_results = self.run(img_path)
                    extracted_value = ocr_results[0] if ocr_results else "N/A"
                    writer.writerow([img_name.replace('_cropped', ''), extracted_value])

        print(f"Resultados salvos em {results_file}")
