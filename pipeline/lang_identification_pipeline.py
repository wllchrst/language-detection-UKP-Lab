import torch
import os
from models.lang_id_classifier import LangIDClassifier
from pipeline.base_pipeline import BasePipeline
from datasets import Dataset, load_dataset
from typing import Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

class LangIdentificationPipeline(BasePipeline):
    """
    A pipeline for language identification specific using classifier that is trained using this class.

    Pipeline without using any training is available on other class within this folder.

    train_np (np means no processed)
    """
    def __init__(self,
                 testing: bool,
                 model_embed_path: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 dataset_path: str = 'papluca/language-identification'):
        super().__init__()
        self.testing = testing
        self.label_encoder = LabelEncoder()
        self.embed_model = SentenceTransformer(model_embed_path)
        self.train_dataset, self.validation_dataset, self.test_dataset = self.process_dataset(
            dataset_path=dataset_path
        )

        self.model = self.train(batch_size=64, epochs=5)

        self.evaluate_training_result(
            model=self.model,
            batch_size=64
        )

    def get_device(self):
        device = None
        if torch.cuda.is_available():
            current_device_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device_index)
            device = torch.device("cuda")
            print(f"PyTorch is currently using GPU: {gpu_name} (Device Index: {current_device_index})")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. PyTorch is running on CPU.")

        return device


    def train(self,
              batch_size: int = 64,
              epochs: int = 5) -> LangIDClassifier:
        device = self.get_device()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

        model = LangIDClassifier(
            input_dimension=len(self.train_dataset[0][0]),
            num_classes=len(self.label_encoder.classes_)
        ).to(device)
        optimizer = torch.optim.Adam(params=model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
            predictions, true_labels = [], []

            with torch.no_grad():
                for X_batch, y_batch in validation_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    total_val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1).cpu().tolist()
                    predictions.extend(preds)
                    true_labels.extend(y_batch.cpu().tolist())

            avg_val_loss = total_val_loss / len(validation_loader)
            val_acc = accuracy_score(true_labels, predictions)

            print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}, "
                  f"val_loss = {avg_val_loss:.4f}, val_acc = {val_acc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        os.makedirs("saved_models", exist_ok=True)
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "saved_models/langid_classifier.pt")
        return model

    def evaluate_training_result(self,
                                 model: LangIDClassifier,
                                 batch_size: int):
        model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        device = self.get_device()
        predictions, true_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                predictions.extend(preds)
                true_labels.extend(y_batch.tolist())

        print('Classification Report')
        print(classification_report(true_labels, predictions))

    def retrieve_dataset(self,
                         dataset_path) -> Tuple[Dataset, Dataset, Dataset]:
        dataset = load_dataset(dataset_path)

        return dataset['train'], dataset['validation'], dataset['test']

    def convert_dataset(self, dataset_np: Dataset) -> TensorDataset:
        texts = dataset_np['text']
        labels = dataset_np['labels']
        embeded_texts = self.embed_model.encode(texts, batch_size=64, show_progress_bar=True)
        labels_encoded = self.label_encoder.fit_transform(labels)

        X = torch.tensor(embeded_texts, dtype=torch.float32)
        y = torch.tensor(labels_encoded, dtype=torch.long)

        return TensorDataset(X, y)

    def process_dataset(self, dataset_path: str) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        train_np, validation_np, test_np = self.retrieve_dataset(dataset_path=dataset_path)

        if self.testing:
            train_np = train_np[:200]
            test_np = test_np[:200]
            validation_np = validation_np[:200]

        train = self.convert_dataset(train_np)
        validation = self.convert_dataset(validation_np)
        test = self.convert_dataset(test_np)

        return train, validation, test

    def predict(self, input: str) -> str:
        return "en"
