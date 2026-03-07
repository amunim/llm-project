# Project Overview

This document summarizes the project goal, data handling, and model choice.

## Goal
Create a privacy-aware, LLM-powered assistant that can answer customer queries using a curated dataset of bank product information.

## Data Handling
- Parse Excel/JSON/CSV/text
- Clean and normalize text
- Apply anonymization (regex masking + optional NER)
- Produce LLM-ready chunks

## Model Choice
Using **Qwen2.5-3B-Instruct** (<= 6B parameters, 3.09B params) for its strong instruction-following behavior, multilingual support, and suitability for prompt engineering and QLoRA fine-tuning via UnSloth.
