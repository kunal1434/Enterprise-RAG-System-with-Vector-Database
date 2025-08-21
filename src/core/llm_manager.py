# LLM Manager with Fine-tuning Support

import os
import torch
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

# Transformers and model handling
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    pipeline, AutoConfig
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM models with fine-tuning capabilities"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fine_tuned_models = {}

        logger.info(f"Initialized LLM Manager with {model_name} on {self.device}")

    def load_model(self) -> bool:
        """Load the base model for inference"""
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                self.model.to(self.device)

            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 256, 
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate response using the loaded model"""
        try:
            if not self.pipeline:
                if not self.load_model():
                    return "Error: Model not loaded"

            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            if outputs and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            else:
                return "No response generated"

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def prepare_training_data(self, data: List[Dict[str, str]]) -> Optional[Dataset]:
        """Prepare data for fine-tuning"""
        if not DATASETS_AVAILABLE:
            logger.error("datasets library not available for fine-tuning")
            return None

        try:
            def tokenize_function(examples):
                texts = []
                for instruction, response in zip(examples['instruction'], examples['response']):
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{self.tokenizer.eos_token}"
                    texts.append(text)

                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )

                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized

            dataset_dict = {
                'instruction': [item['instruction'] for item in data],
                'response': [item['response'] for item in data]
            }

            dataset = Dataset.from_dict(dataset_dict)
            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            return tokenized_dataset

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None

    def fine_tune_model(
        self,
        training_data: List[Dict[str, str]],
        output_dir: str,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """Fine-tune the model using LoRA (if available)"""
        try:
            if not PEFT_AVAILABLE:
                return {
                    'status': 'error',
                    'message': 'PEFT library not available for fine-tuning'
                }

            if not DATASETS_AVAILABLE:
                return {
                    'status': 'error', 
                    'message': 'datasets library not available for fine-tuning'
                }

            if not self.model:
                if not self.load_model():
                    return {'status': 'error', 'message': 'Failed to load base model'}

            # Prepare LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]  # Adjust based on model architecture
            )

            # Apply LoRA to model
            model = get_peft_model(self.model, lora_config)

            # Prepare training data
            train_dataset = self.prepare_training_data(training_data)
            if not train_dataset:
                return {'status': 'error', 'message': 'Failed to prepare training data'}

            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=100,
                learning_rate=learning_rate,
                fp16=self.device == "cuda",
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=2,
                remove_unused_columns=False,
                dataloader_pin_memory=False
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
            )

            # Start training
            logger.info("🚀 Starting fine-tuning...")
            trainer.train()

            # Save the fine-tuned model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            # Store reference
            self.fine_tuned_models[output_dir] = {
                'path': output_dir,
                'base_model': self.model_name,
                'training_data_size': len(training_data),
                'epochs': num_epochs,
                'status': 'completed'
            }

            logger.info(f"✅ Fine-tuning completed. Model saved to {output_dir}")

            return {
                'status': 'success',
                'message': 'Fine-tuning completed successfully',
                'model_path': output_dir,
                'training_samples': len(training_data),
                'epochs': num_epochs
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {
                'status': 'error',
                'message': f'Fine-tuning failed: {str(e)}'
            }

    def load_fine_tuned_model(self, model_path: str) -> bool:
        """Load a fine-tuned model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.to(self.device)

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            logger.info(f"Loaded fine-tuned model from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'pipeline_ready': self.pipeline is not None,
            'fine_tuned_models': list(self.fine_tuned_models.keys()),
            'peft_available': PEFT_AVAILABLE,
            'datasets_available': DATASETS_AVAILABLE
        }

        if self.model:
            try:
                info['model_size'] = sum(p.numel() for p in self.model.parameters())
                info['trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            except:
                pass

        return info

    def save_model_registry(self, file_path: str):
        """Save model registry to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.fine_tuned_models, f, indent=2)

    def load_model_registry(self, file_path: str):
        """Load model registry from file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.fine_tuned_models = json.load(f)
